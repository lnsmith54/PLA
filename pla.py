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
import functools
import os
import json
import time
import string
import random

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from tqdm import trange

from plalib.cta_remixmatch import CTAReMixMatch
from libml import data, utils, augment, ctaugment
import objective as obj_lib
from collections import defaultdict
from third_party import data_util

FLAGS = flags.FLAGS


class AugmentPoolCTACutOut(augment.AugmentPoolCTA):
    @staticmethod
    def numpy_apply_policies(arglist):
        x, cta, probe = arglist
        if x.ndim == 3:
            assert probe
            policy = cta.policy(probe=True)
            return dict(policy=policy,
                        probe=ctaugment.apply(x, policy),
                        image=x)
        assert not probe
        cutout_policy = lambda: cta.policy(probe=False) + [ctaugment.OP('cutout', (1,))]
        return dict(image=np.stack([x[0]] + [ctaugment.apply(y, cutout_policy()) for y in x[1:]]).astype('f'))


class PLA(CTAReMixMatch):
    AUGMENT_POOL_CLASS = AugmentPoolCTACutOut

    def train(self, train_nimg, report_nimg):
        if FLAGS.eval_ckpt:
            self.eval_checkpoint(FLAGS.eval_ckpt)
            return
        batch = FLAGS.batch
        train_labeled = self.dataset.train_labeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_labeled = train_labeled.batch(batch).prefetch(16).make_one_shot_iterator().get_next()
        train_unlabeled = self.dataset.train_unlabeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_unlabeled = train_unlabeled.batch(batch * self.params['uratio']).prefetch(16)
        train_unlabeled = train_unlabeled.make_one_shot_iterator().get_next()
        scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt,
                                                          pad_step_number=10))
        name = ''
        with tf.Session(config=utils.get_config()) as sess:
            self.session = sess
            self.cache_eval()
        with tf.train.MonitoredTrainingSession(
                scaffold=scaffold,
                checkpoint_dir=self.checkpoint_dir,
                config=utils.get_config(),
                save_checkpoint_steps=FLAGS.save_kimg << 10,
                save_summaries_steps=report_nimg - batch) as train_session:
            self.session = train_session._tf_sess()
            gen_labeled = self.gen_labeled_fn(train_labeled)
            gen_unlabeled = self.gen_unlabeled_fn(train_unlabeled)
            self.tmp.step = self.session.run(self.step)
            while self.tmp.step < train_nimg:
                loop = trange(self.tmp.step % report_nimg, report_nimg, batch,
                              leave=False, unit='img', unit_scale=batch,
                              desc='Epoch %d/%d' % (1 + (self.tmp.step // report_nimg), train_nimg // report_nimg))
                for _ in loop:
                    self.train_step(train_session, gen_labeled, gen_unlabeled)
                    while self.tmp.print_queue:
                        loop.write(self.tmp.print_queue.pop(0))
###################### New code
                if FLAGS.boot_flag:
                    FLAGS.boot_flag = False
                    print("Breaking from while in Train: Step= ", self.tmp.step)
                    break
###################### End New code
            while self.tmp.print_queue:
                print(self.tmp.print_queue.pop(0))

        del train_labeled, train_unlabeled, scaffold, train_session
#        print("Return from train")
        return

#    def model(self, batch, lr, wd, wu, mom, confidence, uratio, ema=0.999, **kwargs):
#    def model(self, batch, lr, wd, wu, wclr,  mom, confidence, balance, delT, uratio, clrratio, temperature, ema=0.999, **kwargs):
    def model(self, batch, lr, wd, wu, mom, confidence, balance, delT, uratio, ema=0.999, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        xt_in = tf.placeholder(tf.float32, [batch] + hwc, 'xt')  # Training labeled
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')  # Eval images
        y_in = tf.placeholder(tf.float32, [batch * uratio, 2] + hwc, 'y')  # Training unlabeled (weak, strong)
        l_in = tf.placeholder(tf.int32, [batch], 'labels')  # Labels

#        lrate = tf.clip_by_value(tf.to_float(self.step) / (FLAGS.train_kimg << 10), 0, 1)
#        lr *= tf.cos(lrate * (7 * np.pi) / (2 * 8))
        lrate = tf.clip_by_value(1.5*tf.to_float(self.step) / (FLAGS.train_kimg << 10), 0, 1.5) - 0.5
        lr *= tf.cos(lrate * (0.4965 * np.pi) )
        tf.summary.scalar('monitors/lr', lr)

        # Compute logits for xt_in and y_in
        classifier = lambda x, **kw: self.classifier(x, **kw, **kwargs).logits
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        x = utils.interleave(tf.concat([xt_in, y_in[:, 0], y_in[:, 1]], 0), 2 * uratio + 1)
        logits = utils.para_cat(lambda x: classifier(x, training=True), x)
        logits = utils.de_interleave(logits, 2 * uratio+1)
        post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]
        logits_x = logits[:batch]
        logits_weak, logits_strong = tf.split(logits[batch:], 2)
        del logits, skip_ops

        # Labeled cross-entropy
        loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=l_in, logits=logits_x)
        loss_xe = tf.reduce_mean(loss_xe)
        tf.summary.scalar('losses/xe', loss_xe)

        # Pseudo-label cross entropy for unlabeled data
        pseudo_labels = tf.stop_gradient(tf.nn.softmax(logits_weak))
        loss_xeu = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(pseudo_labels, axis=1),
                                                                  logits=logits_strong)
#        pseudo_mask = tf.to_float(tf.reduce_max(pseudo_labels, axis=1) >= confidence)
        pseudo_mask = self.class_balancing(pseudo_labels, balance, confidence, delT)
        tf.summary.scalar('monitors/mask', tf.reduce_mean(pseudo_mask))
        loss_xeu = tf.reduce_mean(loss_xeu * pseudo_mask)
        tf.summary.scalar('losses/xeu', loss_xeu)

        # L2 regularization
        loss_wd = sum(tf.nn.l2_loss(v) for v in utils.model_vars('classify') if 'kernel' in v.name)
        tf.summary.scalar('losses/wd', loss_wd)

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)

#        train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(
        train_op = tf.train.MomentumOptimizer(lr, mom, use_nesterov=True).minimize(
            loss_xe + wu * loss_xeu + wd * loss_wd, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)
#        print_op = tf.print("Labels ", l_in)
        return utils.EasyDict(
            xt=xt_in, x=x_in, y=y_in, label=l_in, train_op=train_op, # No print_op=print_op,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)))

def main(argv):
    utils.setup_main()
    del argv  # Unused.
#    FLAGS.data_dir = 'data/temp/'
    seedIndx = FLAGS.dataset.find('@')
    seed = int(FLAGS.dataset[seedIndx-1])
    print("dataset name ", FLAGS.dataset)

    dataset = data.PAIR_DATASETS()[FLAGS.dataset]()
#    print(dataset.name, dataset.width, dataset.train_labeled, dataset.train_unlabeled, dataset.test)
    log_width = utils.ilog2(dataset.width)
    model = PLA(
        os.path.join(FLAGS.train_dir, dataset.name, PLA.cta_name()),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        wu=FLAGS.wu,
        mom=FLAGS.mom,
        confidence=FLAGS.confidence,
        balance=FLAGS.balance,
        delT=FLAGS.delT,
        uratio=FLAGS.uratio,
        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
###################### New code
    tic = time.perf_counter()
    train_kimg = FLAGS.train_kimg << 10
    model.train(train_kimg, FLAGS.report_kimg << 10)
    while model.tmp.step < train_kimg:
        sizeIndx = FLAGS.dataset.find('-')
        origSize = int(FLAGS.dataset[seedIndx+1:sizeIndx]) 
        valid = FLAGS.dataset[sizeIndx+1:]
        datasetName = FLAGS.dataset[:FLAGS.dataset.find('.')]
        target = datasetName + '.' + str(seed) + '@'  + str(FLAGS.boot_factor * origSize) + '-' + str(valid)
        print("Target ", target)
        dataset = data.PAIR_DATASETS()[target]()
        log_width = utils.ilog2(dataset.width)
        model.updateDataset(dataset)
        model.train(train_kimg, FLAGS.report_kimg << 10)

    if 'temp' in FLAGS.data_subfolder and  tf.gfile.Exists(data.DATA_DIR+'/'+FLAGS.data_subfolder):
        print("Deleting ",data.DATA_DIR+'/'+FLAGS.data_subfolder)
        tf.compat.v1.gfile.DeleteRecursively(data.DATA_DIR+'/'+FLAGS.data_subfolder)

    elapse = (time.perf_counter() - tic) / 3600
    print(f"Total training time {elapse:0.4f} hours")

###################### End

if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('confidence', 0.95, 'Confidence threshold.')
#    flags.DEFINE_float('lr2', 0.06, 'LR on bootstrap')
    flags.DEFINE_float('wd', 0.0005, 'Weight decay.')
    flags.DEFINE_float('wu', 1, 'Pseudo label loss weight.')
    flags.DEFINE_float('mom', 0.9, 'Momentum coefficient.')
    flags.DEFINE_float('delT', 0.2, 'The amount balance=1 can reduce the confidence threshold.')
    flags.DEFINE_float('min_val_acc', 50, 'The validation set accuracy to trigger bootstrapping.')
    flags.DEFINE_float('imbalance', 0.0, 'Indicator on imbalancing pseudo-labels for weaker classes.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('uratio', 7, 'Unlabeled batch size ratio.')
    flags.DEFINE_integer('boot_factor', 8,'Factor for increasing the number of labeled data')
    flags.DEFINE_integer('balance', 0, 'Method to help balance classes')
    flags.DEFINE_integer('cycling', 0, 'How many epochs to cycle samples (0=no cycling)')
    FLAGS.set_default('augment', 'd.d.d')
    FLAGS.set_default('dataset', 'cifar10.3@250-1')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.03)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)


