#!/usr/bin/env python

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

"""Script to create SSL splits from a dataset.
"""

import json
import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from tqdm import trange, tqdm

from libml import data as libml_data
from libml import utils

flags.DEFINE_integer('seed', 0, 'Random seed to use, 0 for no shuffling.')
flags.DEFINE_integer('size', 0, 'Size of labelled set.')

FLAGS = flags.FLAGS


def get_class(serialized_example):
    return tf.parse_single_example(serialized_example, features={'label': tf.FixedLenFeature([], tf.int64)})['label']

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
        image_feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        return tf.io.parse_single_example(example_proto, image_feature_description)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def main(argv):
    assert FLAGS.size
    argv.pop(0)
    print(argv[1:])
    if any(not tf.gfile.Exists(f) for f in argv[1:]):
        raise FileNotFoundError(argv[1:])
    target = '%s.%d@%d' % (argv[0], FLAGS.seed, FLAGS.size)
    if tf.gfile.Exists(target):
        raise FileExistsError('For safety overwriting is not allowed', target)
    input_files = argv[1:]
    count = 0
    id_class = []
    class_id = defaultdict(list)
    print('Computing class distribution')
    dataset = tf.data.TFRecordDataset(input_files).map(get_class, 4).batch(1 << 10)
    it = dataset.make_one_shot_iterator().get_next()
    try:
        with tf.Session() as session, tqdm(leave=False) as t:
            while 1:
                old_count = count
                for i in session.run(it):
                    id_class.append(i)
                    class_id[i].append(count)
                    count += 1
                t.update(count - old_count)
    except tf.errors.OutOfRangeError:
        pass
    print('%d records found' % count)
    nclass = len(class_id)
    print("Labels ", id_class[:20])

    assert min(class_id.keys()) == 0 and max(class_id.keys()) == (nclass - 1)
    train_stats = np.array([len(class_id[i]) for i in range(nclass)], np.float64)
    train_stats /= train_stats.max()
    if 'stl10' in argv[1]:
        # All of the unlabeled data is given label 0, but we know that
        # STL has equally distributed data among the 10 classes.
        train_stats[:] = 1

    print('  Stats', ' '.join(['%.2f' % (100 * x) for x in train_stats]))
    assert min(class_id.keys()) == 0 and max(class_id.keys()) == (nclass - 1)
    class_id = [np.array(class_id[i], dtype=np.int64) for i in range(nclass)]
    if FLAGS.seed:
        np.random.seed(FLAGS.seed)
        for i in range(nclass):
            np.random.shuffle(class_id[i])

    # Distribute labels to match the input distribution.
    npos = np.zeros(nclass, np.int64)
    label = []
    for i in range(FLAGS.size):
        c = np.argmax(train_stats - npos / max(npos.max(), 1))
        label.append(class_id[c][npos[c]])
        npos[c] += 1

    del npos, class_id
    print(' Classes', ' '.join(['%d' %  id_class[x] for x in label]))
    label = frozenset([int(x) for x in label])
#    print(label)
#    exit(1)
    if 'stl10' in argv[1] and FLAGS.size == 1000:
        data = tf.gfile.Open(os.path.join(libml_data.DATA_DIR, 'stl10_fold_indices.txt'), 'r').read()
        label = frozenset(list(map(int, data.split('\n')[FLAGS.seed].split())))

    print('Creating split in %s' % target)
    tf.gfile.MakeDirs(os.path.dirname(target))
    with tf.python_io.TFRecordWriter(target + '-label.tfrecord') as writer_label:
        pos, loop = 0, trange(count, desc='Writing records')
        for input_file in input_files:
            for record in tf.python_io.tf_record_iterator(input_file):
#            for record in tf.compat.v1.io.tf_record_iterator(input_file):
#                if pos in label:
#            example = tf.train.Example()
#            for record in tf.data.TFRecordDataset(input_file).take(1):
#                parsed_image_dataset = record.map(_parse_image_function)
#                print(len(record))
#                print(tf.train.Example.FromString(record))
                print(tf.train.Example.FromString(record).features.feature['label'].int64_list.value[0])
#                tf.train.Example.FromString(record).features.feature['label'].int64_list.value[0] = 10
#                feat = dict(image=_bytes_feature(tf.train.Example.FromString(record).features.feature['image'].bytes_list.value[0]),
#                            label=_int64_feature(10))
#                newrecord = tf.train.Example(features=tf.train.Features(feature=feat))
#                print(newrecord)
#                print(len(newrecord.SerializeToString()))
#                print(newrecord.features.feature['label'].int64_list.value[0])
#                writer.write(record.SerializeToString())
                if pos > 20:
                    exit(1)
#                print(tf.io.decode_raw(record,tf.int32))

#                print(parsed_image_dataset)
#                for image_features in parsed_image_dataset:
#                    label = image_features['labels'].numpy()
#                    print(label)
#                print(example.ParseFromString(record))
#                print(example.ParseFromString(record.numpy()))
#            for record in tf.data.TFRecordDataset(input_file).make_one_shot_iterator().get_next():
                if pos in label:
#                    print(record.features.feature['label'].int64_list.value[0])
                    writer_label.write(record)
                pos += 1
                loop.update()
        loop.close()
    with tf.gfile.Open(target + '-label.json', 'w') as writer:
        writer.write(json.dumps(dict(distribution=train_stats.tolist(), label=sorted(label)), indent=2, sort_keys=True))



if __name__ == '__main__':
    utils.setup_tf()
    app.run(main)

#        train_labeled = self.dataset.train_labeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
#        train_labeled = train_labeled.batch(batch).prefetch(16).make_one_shot_iterator().get_next()
def _save_as_tfrecord(data, filename):
    assert len(data['images']) == len(data['labels'])
    filename = os.path.join(libml_data.DATA_DIR, filename + '.tfrecord')
    print('Saving dataset:', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for x in trange(len(data['images']), desc='Building records'):
            feat = dict(image=_bytes_feature(data['images'][x]),
                        label=_int64_feature(data['labels'][x]))
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
    print('Saved:', filename)

