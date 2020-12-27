# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf
import numpy as np
from absl import flags
import statistics 
import os
import string
import random
import json, math

from fully_supervised.lib.train import ClassifyFullySupervised
from libml import data
from libml.augment import AugmentPoolCTA
from libml.ctaugment import CTAugment
from libml.train import ClassifySemi
from tqdm import trange, tqdm
from collections import defaultdict

FLAGS = flags.FLAGS

flags.DEFINE_integer('adepth', 2, 'Augmentation depth.')
flags.DEFINE_float('adecay', 0.99, 'Augmentation decay.')
flags.DEFINE_float('ath', 0.80, 'Augmentation threshold.')
flags.DEFINE_boolean('boot_flag', False,'Binary flag for bootstrapping')


class CTAClassifySemi(ClassifySemi):
    """Semi-supervised classification."""
    AUGMENTER_CLASS = CTAugment
    AUGMENT_POOL_CLASS = AugmentPoolCTA

    @classmethod
    def cta_name(cls):
        return '%s_depth%d_th%.2f_decay%.3f' % (cls.AUGMENTER_CLASS.__name__,
                                                FLAGS.adepth, FLAGS.ath, FLAGS.adecay)

    def __init__(self, train_dir: str, dataset: data.DataSets, nclass: int, **kwargs):
        ClassifySemi.__init__(self, train_dir, dataset, nclass, **kwargs)
        self.augmenter = self.AUGMENTER_CLASS(FLAGS.adepth, FLAGS.ath, FLAGS.adecay)
        self.best_acc=0
        self.best_accStd=0
        self.counter=0
        seedIndx = FLAGS.dataset.find('@')
        sizeIndx = FLAGS.dataset.find('-')
        self.origSize = int(FLAGS.dataset[seedIndx+1:sizeIndx]) 
        self.boot = False

    def updateKeywords(self, **kwargs):
#        print("Old arguements")
#        for k, v in sorted(self.kwargs.items()):
#            print('%-32s %s' % (k, v))
        print("New arguements")
        for k, v in sorted(kwargs.items()):
            self.kwargs[k] = v
            print('%-32s %s' % (k, v))
        print("updated arguements")
        for k, v in sorted(self.kwargs.items()):
            print('%-32s %s' % (k, v))
#        self.ops = self.model(**self.kwargs)

    def updateDataset(self, dataset):
        self.dataset = dataset
        print("New dataset name ", dataset.name)

    def gen_labeled_fn(self, data_iterator):
        def wrap():
            batch = self.session.run(data_iterator)
            batch['cta'] = self.augmenter
            batch['probe'] = True
            return batch

        return self.AUGMENT_POOL_CLASS(wrap)

    def gen_unlabeled_fn(self, data_iterator):
        def wrap():
            batch = self.session.run(data_iterator)
            batch['cta'] = self.augmenter
            batch['probe'] = False
            return batch

        return self.AUGMENT_POOL_CLASS(wrap)

    def train_step(self, train_session, gen_labeled, gen_unlabeled):
        #        try:
        x, y = gen_labeled(), gen_unlabeled()
        v = train_session.run([self.ops.classify_op, self.ops.train_op, self.ops.update_step],
                              feed_dict={self.ops.y: y['image'],
                                         self.ops.x: x['probe'],
                                         self.ops.xt: x['image'],
                                         self.ops.label: x['label']})
        self.tmp.step = v[-1]
        lx = v[0]
        for p in range(lx.shape[0]):
            error = lx[p]
            error[x['label'][p]] -= 1
            error = np.abs(error).sum()
            self.augmenter.update_rates(x['policy'][p], 1 - 0.5 * error)

    def cache_eval(self):
        """Cache datasets for computing eval stats."""

        def collect_samples(dataset, name):
            """Return numpy arrays of all the samples from a dataset."""
#            pbar = tqdm(desc='Caching %s examples' % name)
            it = dataset.batch(1).prefetch(16).make_one_shot_iterator().get_next()
            images, labels = [], []
            while 1:
                try:
                    v = self.session.run(it)
                except tf.errors.OutOfRangeError:
                    break
                images.append(v['image'])
                labels.append(v['label'])
#                pbar.update()

            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels, axis=0)
#            pbar.close()
            return images, labels

        if 'test' not in self.tmp.cache:
            self.tmp.cache.test = collect_samples(self.dataset.test.parse(), name='test')
            self.tmp.cache.valid = collect_samples(self.dataset.valid.parse(), name='valid')
            self.tmp.cache.train_labeled = collect_samples(self.dataset.train_labeled.take(10000).parse(),
                                                           name='train_labeled')
            self.tmp.cache.train_original = collect_samples(self.dataset.train_original.parse(),
                                                           name='train_original')

    def eval_stats(self, batch=None, feed_extra=None, classify_op=None, verbose=True):
        """Evaluate model on train, valid and test."""
        batch = batch or FLAGS.batch
        classify_op = self.ops.classify_op if classify_op is None else classify_op
        epochs = self.tmp.step // 65536
        accuracies = []
        for subset in ('train_labeled', 'valid', 'test'):
            images, labels = self.tmp.cache[subset]
            predicted = []
            for x in range(0, images.shape[0], batch):
                p = self.session.run(
                    classify_op,
                    feed_dict={
                        self.ops.x: images[x:x + batch],
                        **(feed_extra or {})
                    })
                predicted.append(p)
            predicted = np.concatenate(predicted, axis=0)
            pred = predicted.argmax(1)
            probs = predicted.max(1)
            accuracies.append((pred == labels).mean() * 100)
#####  New Code 
            if verbose and  subset == 'valid':
                val_class_acc = []
                for c in range(self.nclass):
                    mask = (labels == c)
                    val_class_acc.append((pred[mask] == c).mean() *100)
                print("Validation set class accuracies ",accuracies[1],  val_class_acc)
#                if accuracies[1] > FLAGS.min_val_acc and not self.boot:
                if min(val_class_acc) > FLAGS.min_val_acc and not self.boot:
                    currentSize = int(FLAGS.boot_factor * self.origSize)
                    print("First bootstrap episode: currentSize = ", currentSize)
                    self.bootstrap(currentSize)
                    self.boot = True
                    FLAGS.boot_flag = True
#                    self.kwargs['lr'] = FLAGS.lr2
                elif self.boot and FLAGS.cycling > 0:
                    if epochs % FLAGS.cycling == 0:
                        currentSize = int(FLAGS.boot_factor * self.origSize)
                        print("Sample Cycling at epoch ", epochs, "  currentSize = ", currentSize)
                        self.bootstrap(currentSize)
                        FLAGS.boot_flag = True
            elif verbose and  subset == 'test':
                test_class_acc = []
                for c in range(self.nclass):
                    mask = (labels == c)
                    test_class_acc.append((pred[mask] == c).mean() *100)
                print("Test set class accuracies ",accuracies[1],  test_class_acc)

        testAcc = float(accuracies[2])
        if testAcc  > self.best_acc:
            self.best_acc = testAcc

        if verbose:
            lrate = 1.5*float(self.tmp.step) / (FLAGS.train_kimg << 10) - 0.5
            lr = FLAGS.lr * math.cos(lrate * 0.4965 * np.pi)
            acc = []
            acc.append(epochs)
            acc.append(self.tmp.step >> 10)
            acc.append(lr)
            for item in accuracies:
                acc.append(item) 
            acc.append(self.best_acc)
#            print(acc)
            tup = tuple(acc)
#            self.train_print('Epochs %d, kimg %-5d accuracy train/valid/test/best_test  %.2f  %.2f  %.2f  %.2f  ' % tup)
            self.train_print('Epochs %d, kimg %-5d lr %.3f  accuracy train/valid/test/best_test  %.2f  %.2f  %.2f  %.2f  ' % tup)

        return np.array(accuracies, 'f')

    def get_class(self, serialized_example):
        return tf.parse_single_example(serialized_example, features={'label': tf.FixedLenFeature([], tf.int64)})['label']

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def get_random_string(self, length):
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        #    print("Random string of length", length, "is:", result_str)
        return result_str

    def bootstrap(self, currentSize ):
        """Output the highest confidence pseudo-labeled examples."""

        # Read in the original labeled data sample numbers
        DATA_DIR = FLAGS.data_dir or os.environ['ML_DATA']
#        print("DATA_DIR ", DATA_DIR)
        indx = FLAGS.dataset.index('-')
        name = FLAGS.dataset[:indx] + '-label.json'
        root = os.path.join(DATA_DIR, 'SSL2', name)
        with open(root) as f:
            origSamples = json.load(f)
        classify_op = self.ops.classify_op 
        images, labels = self.tmp.cache['train_original']

        batch = FLAGS.batch # len(labels)//10
        predicted = []
        for x in range(0, images.shape[0], batch):
            p = self.session.run(
                classify_op,
                feed_dict={
                    self.ops.x: images[x:x + batch]
                })
            predicted.append(p)
        predicted = np.concatenate(predicted, axis=0)
        preds = predicted.argmax(1)
        probs = predicted.max(1)
        top = np.argsort(-probs,axis=0)
#        print("preds  ", preds.shape, preds)
#        print("probs  ", probs.shape, probs)
#        unique_train_counts = [0]*self.nclass
        unique_train_pseudo_labels, unique_train_counts = np.unique(preds, return_counts=True)
        print("Number of training pseudo-labels in each class: ", unique_train_counts," for classes: ", unique_train_pseudo_labels)
        numPerClass = currentSize // self.nclass
        sortByClass = np.random.randint(0,high=len(labels), size=(self.nclass, numPerClass), dtype=int)
        indx = np.zeros([self.nclass], dtype=int)
        matches = np.zeros([self.nclass, numPerClass], dtype=int)
        labls  = preds[top]
        samples = top
        trainLabelAcc = 0
#        print("number of samples ", len(top))
#        print("labls",labls[:100])
#        print("labels",labels[top[:100]])
#        print("pseudo-labels",preds[top[:100]])
#        print("probs",probs[top[:100]])
#        print("samples", top[:100])
        for i in origSamples['label']:
            y = labels[i]
            sortByClass[y, indx[y]] = i
            matches[y, indx[y]] = 1
            if preds[i] == y:
                trainLabelAcc += 1
            indx[y] += 1
        trainLabelAcc = 100 * trainLabelAcc / len(origSamples['label'])
        for i in range(len(top)):
            if indx[labls[i]] < numPerClass:
                sortByClass[labls[i], indx[labls[i]]] = samples[i]
#                print(i, labls[i], labels[top[i]],indx[labls[i]])
                if labls[i] == labels[top[i]]:
                    matches[labls[i], indx[labls[i]]] = 1
                indx[labls[i]] += 1

        if min(indx) < numPerClass:
            print("Counts of at least one class ", indx, " is lower than ", numPerClass)
#        print("SortByClass = ", sortByClass)
        print("matches",matches[:self.nclass, :numPerClass])

        # Set up new dataset in a random folder
        datasetName = self.dataset.name[:self.dataset.name.find('.')]
        if not self.boot:
            letters = string.ascii_letters
            subfolder = ''.join(random.choice(letters) for i in range(8))
            FLAGS.data_subfolder = 'temp/' + subfolder
            tf.gfile.MakeDirs(data.DATA_DIR+'/temp/'+subfolder)
            if not tf.gfile.Exists(data.DATA_DIR+'/temp/'+subfolder+'/'+datasetName+'-unlabel.json'):
                infile = data.DATA_DIR+'/SSL2/'+datasetName+'-unlabel.'
                outfile = data.DATA_DIR+'/temp/'+subfolder+'/'+datasetName+'-unlabel.'
                print("Copied from ",infile, "* to ", outfile +'*')
                tf.io.gfile.copy(infile+'json', outfile + 'json')
                tf.io.gfile.copy(infile+'tfrecord', outfile + 'tfrecord')
            
        seedIndx = FLAGS.dataset.find('@')
        seed = int(FLAGS.dataset[seedIndx-1])
        target =  '%s.%s@%d.npy' % (datasetName,seed, currentSize)
        target = '%s/%s/%s' % (data.DATA_DIR, FLAGS.data_subfolder, target)
        if tf.gfile.Exists(target):
            prevSortByClass = np.load(target)
            count = 0
            for j in range(self.origSize+1,numPerClass):
                for i in range(self.nclass):
                    if prevSortByClass[i,j] in sortByClass:
                        count += 1
            repeats = count/( (numPerClass-self.origSize)*self.nclass)
            print("Percentage of repeat pseudo-labeled samples ", repeats)
        print("Saving ", target)
        np.save(target, sortByClass[0:self.nclass, :numPerClass])

        classAcc = 100*np.sum(matches, axis=1)/numPerClass
        print("Accuracy of the predicted pseudo-labels: top ", numPerClass,  ", ", np.mean(classAcc), classAcc )

        input_file=data.DATA_DIR+'/'+datasetName+'-train.tfrecord'
        target = ('%s/%s/%s.%d@%d' % (data.DATA_DIR, FLAGS.data_subfolder, datasetName, seed, currentSize) )
        print("input_file ", input_file," target ",target)
        if tf.gfile.Exists(target + '-label.tfrecord'): 
            tf.io.gfile.remove(target + '-label.tfrecord')
            tf.io.gfile.remove(target + '-label.json')

        count = 0
        id_class = []
        class_id = defaultdict(list)
#        print('Computing class distribution')
        dataset = tf.data.TFRecordDataset(input_file).map(self.get_class, 4).batch(1 << 10)
        it = dataset.make_one_shot_iterator().get_next()
        try:
            with tf.Session() as session:
                while 1:
#                    old_count = count
                    for i in session.run(it):
                        id_class.append(i)
                        class_id[i].append(count)
                        count += 1
#                    t.update(count - old_count)
        except tf.errors.OutOfRangeError:
            pass
#        print('%d records found' % count)
        nclass = len(class_id)
        assert min(class_id.keys()) == 0 and max(class_id.keys()) == (nclass - 1)
        train_stats = np.array([len(class_id[i]) for i in range(nclass)], np.float64)
        train_stats /= train_stats.max()
        if 'stl10' in self.train_dir:
            # All of the unlabeled data is given label 0, but we know that
            # STL has equally distributed data among the 10 classes.
            train_stats[:] = 1

#        print('  Stats', ' '.join(['%.2f' % (100 * x) for x in train_stats]))
        assert min(class_id.keys()) == 0 and max(class_id.keys()) == (nclass - 1)
        class_id = [np.array(class_id[i], dtype=np.int64) for i in range(nclass)]

        # Distribute labels to match the input distribution.
        train_data = sortByClass[0:self.nclass, :numPerClass]
        samples = []
        pseudo_labels = []
        for i in range(currentSize):
            c = i%nclass
            npos = i//nclass
            pseudo_labels.append(c)
            samples.append(train_data[c][npos])
#            print("Label ",id_class[train_data[c][npos]]," Pseudo-label ", c, " Top confidence ", npos, " train_data ", train_data[c][npos])

        del train_data, it, dataset
        label = frozenset([int(x) for x in samples])
        if 'stl10' in self.train_dir and currentSize == 1000:
            train_data = tf.gfile.Open(os.path.join(data.DATA_DIR, 'stl10_fold_indices.txt'), 'r').read()
            label = frozenset(list(map(int, data.split('\n')[seed].split())))

        tf.gfile.MakeDirs(os.path.dirname(target))
        with tf.python_io.TFRecordWriter(target + '-label.tfrecord') as writer_label:
            pos, loop = 0, trange(count, desc='Writing records')
            #for infile in input_file:
            for record in tf.python_io.tf_record_iterator(input_file):
                if pos in label:
                    pseudo_label = pseudo_labels[samples.index(pos)]
                    feat = dict(image=self._bytes_feature(tf.train.Example.FromString(record).features.feature['image'].bytes_list.value[0]),
                                label=self._int64_feature(pseudo_label))
                    newrecord = tf.train.Example(features=tf.train.Features(feature=feat))
                    writer_label.write(newrecord.SerializeToString())
                pos += 1
                loop.update()
            loop.close()
        with tf.gfile.Open(target + '-label.json', 'w') as writer:
            writer.write(json.dumps(dict(distribution=train_stats.tolist(), label=sorted(label)), indent=2, sort_keys=True))
        return

####################### Modification 
    def class_balancing(self, pseudo_labels, balance, confidence, delT):

        if balance > 0:
            pLabels = tf.math.argmax(pseudo_labels, axis=1)
            pLabels = tf.cast(pLabels,dtype=tf.float32)
            classes, idx, counts = tf.unique_with_counts(pLabels)
            shape = tf.constant([self.dataset.nclass])
            classes = tf.cast(classes,dtype=tf.int32)
            class_count = tf.scatter_nd(tf.reshape(classes,[tf.size(classes),1]),counts, shape)

            class_count = tf.cast(class_count,dtype=tf.float32)
            mxCount = tf.reduce_max(class_count, axis=0)

            pLabels = tf.cast(pLabels,dtype=tf.int32)
            if balance == 1 or balance == 4:
                confidences = tf.zeros_like(pLabels,dtype=tf.float32)
                ratios  = 1.0 - tf.math.divide_no_nan(class_count, mxCount)
                ratios  = confidence - delT*ratios
                confidences = tf.gather_nd(ratios, tf.reshape(pLabels,[tf.size(pLabels),1]) )
                pseudo_mask = tf.reduce_max(pseudo_labels, axis=1) >= confidences
            else:
                pseudo_mask = tf.reduce_max(pseudo_labels, axis=1) >= confidence

            if balance == 3 or balance == 4:
                classes, idx, counts = tf.unique_with_counts(tf.boolean_mask(pLabels,pseudo_mask))
                shape = tf.constant([self.dataset.nclass])
                classes = tf.cast(classes,dtype=tf.int32)
                class_count = tf.scatter_nd(tf.reshape(classes,[tf.size(classes),1]),counts, shape)
                class_count = tf.cast(class_count,dtype=tf.float32)
            pseudo_mask = tf.cast(pseudo_mask,dtype=tf.float32)

            if balance > 1:
                ratios  = tf.math.divide_no_nan(tf.ones_like(class_count,dtype=tf.float32),class_count)
                ratio = tf.gather_nd(ratios, tf.reshape(pLabels,[tf.size(pLabels),1]) )
                Z = tf.reduce_sum(pseudo_mask)
                pseudo_mask = tf.math.multiply(pseudo_mask, tf.cast(ratio,dtype=tf.float32))
                pseudo_mask = tf.math.divide_no_nan(tf.scalar_mul(Z, pseudo_mask), tf.reduce_sum(pseudo_mask))
        else:
            pseudo_mask = tf.cast(tf.reduce_max(pseudo_labels, axis=1) >= confidence,dtype=tf.float32)

        return pseudo_mask 

###################### End

##### End of new code

class CTAClassifyFullySupervised(ClassifyFullySupervised, CTAClassifySemi):
    """Fully-supervised classification."""

    def train_step(self, train_session, gen_labeled):
        x = gen_labeled()
        v = train_session.run([self.ops.classify_op, self.ops.train_op, self.ops.update_step],
                              feed_dict={self.ops.x: x['probe'],
                                         self.ops.xt: x['image'],
                                         self.ops.label: x['label']})
        self.tmp.step = v[-1]
        lx = v[0]
        for p in range(lx.shape[0]):
            error = lx[p]
            error[x['label'][p]] -= 1
            error = np.abs(error).sum()
            self.augmenter.update_rates(x['policy'][p], 1 - 0.5 * error)
