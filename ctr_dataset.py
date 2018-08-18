# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Prepare MovieLens dataset for wide-deep."""

import functools
import os
import numpy as np
import tensorflow as tf
import collections
import pickle
from datetime import datetime

_RAW_COLUMN_NAMES = \
                ['id',          'click',            'hour',         'C1',           'banner_pos',           'site_id',
                 'site_domain', 'site_category',    'app_id',       'app_domain',   'app_category',         'device_id',
                 'device_ip',   'device_model',     'device_type',  'device_conn_type', 'C14',              'C15',
                 'C16',         'C17',              'C18',          'C19',          'C20',                  'C21']
_COLUMN_NAMES = _RAW_COLUMN_NAMES.copy()
_COLUMN_NAMES.append('weekday')

_BUFFER_SUBDIR = "wide_deep_buffer"
_FEATURE_MAP = {
    column_name: tf.FixedLenFeature([1], dtype=tf.int64) for column_name in _COLUMN_NAMES[1:]
}

_COLUMN_DIM = \
                [0,             2,                  24,             7,              7,                      4642,
                 7564,          26,                 8291,           548,            36,                     2484613,
                 6134351,       8162,               5,              4,              2470,                   8,
                 9,             407,                4,              66,             172,                    55,
                 7]

_COLUMN_EMBEDDING_DIM = \
                [0,             2,                  24,             7,              7,                      10,
                 10,           26,                 10,           10,             36,                     10,
                 10,           10,                5,              4,              10,                    8,
                 9,             10,                4,              66,             100,                    55,
                 7]

_CSV_COLUMN_DEFAULTS = [[0]]*len(_COLUMN_DIM)
_CSV_COLUMN_DEFAULTS[0] = ['']

tf.app.flags.DEFINE_string("train_filename", 'data/train.csv', "Training data. E.g., train.csv.")
tf.app.flags.DEFINE_boolean("train_valid_split", False, "If True, will split 30s day as valid data set")
tf.app.flags.DEFINE_string("train_split_filename", 'train_split.csv', "split train file name")
tf.app.flags.DEFINE_string("valid_split_filename", 'valid_split.csv', "split valid file name")
tf.app.flags.DEFINE_string("output_dir", 'output', "processed data output directory")
tf.app.flags.DEFINE_string("feature_count_filename", 'feature_count.pickle', "split valid file name")
tf.app.flags.DEFINE_string("feature_map_filename", 'feature_map.pickle', "split valid file name")
tf.app.flags.DEFINE_string("train_format_filename", 'train_format.tfrecord', "split valid file name")
tf.app.flags.DEFINE_string("valid_format_filename", 'valid_format.tfrecord', "split valid file name")
tf.app.flags.DEFINE_boolean("gen_csv", True, "If True, will split 30s day as valid data set")
tf.app.flags.DEFINE_string("train_csv_filename", 'train_format.csv', "split valid file name")
tf.app.flags.DEFINE_string("valid_csv_filename", 'valid_format.csv', "split valid file name")
tf.app.flags.DEFINE_boolean('cal_md5sum', False, 'md5sum')
tf.app.flags.DEFINE_boolean('gen_valid_dataset', False, 'md5sum')
tf.app.flags.DEFINE_boolean('build_feature_map', False, 'md5sum')
tf.app.flags.DEFINE_enum('gen_type', 'valid', ['train', 'valid', 'all'], 'gen type')

FLAGS = tf.app.flags.FLAGS


def train_valid_split(train_path, train_split_path, valid_split_path):
    log_prefix = 'train_valid_split'
    tf.logging.info('{} ...'.format(log_prefix))
    with open(train_path, 'rb') as ftrain, \
            open(train_split_path, 'wb') as ftrain_split, \
            open(valid_split_path, 'wb') as fvalid_split:
        header = ftrain.readline()
        ftrain_split.write(header)
        fvalid_split.write(header)
        for line_no, line in enumerate(ftrain):
            timestamp = line.split(b',')[2]
            assert len(timestamp) == 8, timestamp
            assert timestamp[0:2] == b'14', timestamp
            assert timestamp[2:4] == b'10', timestamp
            if timestamp[4:6] != b'30':
                ftrain_split.write(line)
            else:
                fvalid_split.write(line)

            tf.logging.log_every_n(tf.logging.INFO, '{}: Progress {}'.format(log_prefix, line_no), 400000)
        tf.logging.info('{}: Progress {}, done!!!'.format(log_prefix, line_no))


def get_feature_map(train_path, feature_count_path, feature_map_path):
    log_prefix = 'get_feature_map'
    tf.logging.info('{} (skip feature columns id, clicked, hour)...'.format(log_prefix))

    count_list = [collections.defaultdict(lambda: 0) for _ in range(len(_RAW_COLUMN_NAMES))]
    with open(train_path, 'r') as ftrain:
        ftrain.readline()
        for line_no, line in enumerate(ftrain):
            columns = line.strip().split(',')
            for index, feature in enumerate(columns):
                if index > 2:
                    count_list[index][feature] += 1
            tf.logging.log_every_n(tf.logging.INFO, '{}: count Progress {}'.format(log_prefix, line_no), 400000)
        tf.logging.info('{}: count Progress {}, done!!!'.format(log_prefix, line_no))

    count_list = list(map(lambda x: dict(x), count_list))
    with open(feature_count_path, 'wb') as fcount:
        pickle.dump(count_list, fcount, pickle.HIGHEST_PROTOCOL)

    tf.logging.info('{} build feature map ...'.format(log_prefix))
    map_list = [dict() for _ in range(len(_RAW_COLUMN_NAMES))]
    for index, feature_count in enumerate(count_list):
        for num, feature_value in enumerate(feature_count):
            map_list[index][feature_value] = num
    with open(feature_map_path, 'wb') as fmap:
        pickle.dump(map_list, fmap, pickle.HIGHEST_PROTOCOL)
    tf.logging.info('{} build feature map done!!!'.format(log_prefix))

    return map_list


def build_example(line, with_id=False):
    feature_dict = {feature_name: tf.train.Feature(int64_list=tf.train.Int64List(value=[feature]))
                    for feature_name, feature in zip(_COLUMN_NAMES[1:], line[1:])}

    if with_id:
        feature_dict['id'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[line[0].encode()]))
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def gen_train_dataset(train_path, format_path, map_list, csv_path='train_format.csv', gen_csv=True):
    log_prefix = 'convert_origin_file'
    tf.logging.info('{} ...'.format(log_prefix))

    if gen_csv:
        fcsv = open(csv_path, 'w')
    with open(train_path, 'r') as ftrain, tf.python_io.TFRecordWriter(format_path) as writer:
        tmp = ftrain.readline().strip().split(',')
        tmp.append('weekday')
        if gen_csv:
            fcsv.write(','.join(tmp) + '\n')
        for line_no, line in enumerate(ftrain):
            for index, feature in enumerate(line.strip().split(',')):
                if index > 2:
                    tmp[index] = map_list[index][feature]
                elif index == 2:
                    parsed_time = datetime.strptime('20' + feature, '%Y%m%d%H')
                    tmp[index] = parsed_time.hour
                elif index == 1:
                    tmp[index] = int(feature)
                elif index == 0:
                    tmp[index] = feature
            # add week info
            tmp[-1] = parsed_time.weekday()
            example = build_example(tmp)
            writer.write(example.SerializeToString())
            if gen_csv:
                fcsv.write(','.join(map(lambda x: str(x), tmp)) + '\n')

            tf.logging.log_every_n(tf.logging.INFO, '{}: Progress {}'.format(log_prefix, line_no), 400000)
        tf.logging.info('{}: Progress {}, done!!!'.format(log_prefix, line_no))


def gen_valid_dataset(train_path, format_path, map_list, csv_path='valid_format.csv', gen_csv=True):
    log_prefix = 'gen_valid_dataset'
    tf.logging.info('{} ...'.format(log_prefix))

    if gen_csv:
        fcsv = open(csv_path, 'w')
    with open(train_path, 'r') as ftrain, tf.python_io.TFRecordWriter(format_path) as writer:
        tmp = ftrain.readline().strip().split(',')
        tmp.append('weekday')
        if gen_csv:
            fcsv.write(','.join(tmp) + '\n')

        skip_line = 0
        for line_no, line in enumerate(ftrain):
            try:
                for index, feature in enumerate(line.strip().split(',')):
                    if index > 2:
                        tmp[index] = map_list[index][feature]
                    elif index == 2:
                        parsed_time = datetime.strptime('20' + feature, '%Y%m%d%H')
                        tmp[index] = parsed_time.hour
                    elif index == 1:
                        tmp[index] = int(feature)
                    elif index == 0:
                        tmp[index] = feature
                # add week info
                tmp[-1] = parsed_time.weekday()
                example = build_example(tmp)
                writer.write(example.SerializeToString())
                if gen_csv:
                    fcsv.write(','.join(map(lambda x: str(x), tmp)) + '\n')
            except KeyError as e:
                skip_line += 1

            tf.logging.log_every_n(tf.logging.INFO, '{}: Progress {}'.format(log_prefix, line_no), 400000)
        tf.logging.info('{}: Progress {}, skip {}, done!!!'.format(log_prefix, line_no, skip_line))


def build_model_columns():
    ctr_columns = [tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(feature_name, _COLUMN_DIM[index]),
        _COLUMN_EMBEDDING_DIM[index], max_norm=np.sqrt(_COLUMN_EMBEDDING_DIM[index]))
        for index, feature_name in enumerate(_COLUMN_NAMES) if index > 1]

    return ctr_columns


def _deserialize(examples_serialized):
    features = tf.parse_example(examples_serialized, _FEATURE_MAP)
    #print(features)
    label = features.pop('click')
    return features, label


def parse_csv(value):
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS, use_quote_delim=False)
    features = dict(zip(_COLUMN_NAMES, columns))
    features.pop('id')
    labels = features.pop('click')
    return features, [labels]


def get_input_fn(train_path, batch_size, repeat, shuffle, use_tfrecord=False):
    def csv_input_fn():
        # Extract lines from input files using the Dataset API.
        dataset = tf.data.TextLineDataset(train_path)
        dataset = dataset.skip(1)
        dataset = dataset.map(parse_csv, num_parallel_calls=5)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle)



        # We call repeat after shuffling, rather than before, to prevent separate
        # epochs from blending together.
        dataset = dataset.repeat(repeat)
        dataset = dataset.batch(batch_size)
        return dataset

    def tfrecord_input_fn():
        dataset = tf.data.TFRecordDataset(train_path)
        # batch comes before map because map can deserialize multiple examples.
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(_deserialize, num_parallel_calls=8)
        if shuffle:
            dataset = dataset.shuffle(shuffle)

        dataset = dataset.repeat(repeat)
        return dataset.prefetch(1)

    if use_tfrecord:
        return tfrecord_input_fn
    else:
        return csv_input_fn


def main(_):
    train_path = FLAGS.train_filename
    train_split_path= os.path.join(FLAGS.output_dir, FLAGS.train_split_filename)
    valid_split_path = os.path.join(FLAGS.output_dir, FLAGS.valid_split_filename)
    feature_count_path = os.path.join(FLAGS.output_dir, FLAGS.feature_count_filename)
    feature_map_path = os.path.join(FLAGS.output_dir, FLAGS.feature_map_filename)
    train_format_path = os.path.join(FLAGS.output_dir, FLAGS.train_format_filename)
    valid_format_path = os.path.join(FLAGS.output_dir, FLAGS.valid_format_filename)
    train_csv_path = os.path.join(FLAGS.output_dir, FLAGS.train_csv_filename)
    valid_csv_path = os.path.join(FLAGS.output_dir, FLAGS.valid_csv_filename)
    print(FLAGS)
    return
    if FLAGS.train_valid_split:
        train_valid_split(train_path, train_split_path, valid_split_path)
        train_path = train_split_path

    if FLAGS.build_feature_map:
        map_list = get_feature_map(train_path, feature_count_path, feature_map_path)
    else:
        with open(feature_map_path, 'rb') as fmap:
            map_list = pickle.load(fmap)
    if FLAGS.gen_type == 'all' or FLAGS.gen_type == 'train':
        gen_train_dataset(train_path, train_format_path, map_list, train_csv_path, FLAGS.gen_csv)
        if FLAGS.cal_md5sum:
            os.system('md5sum ' + train_format_path)
            if FLAGS.gen_csv:
                os.system('md5sum ' + train_csv_path)
    if FLAGS.gen_type == 'all' or FLAGS.gen_type == 'valid':
        gen_valid_dataset(valid_split_path, valid_format_path, map_list, valid_csv_path, FLAGS.gen_csv)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
