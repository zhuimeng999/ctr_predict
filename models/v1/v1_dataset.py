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

import numpy as np
import tensorflow as tf

COLUMN_NAMES = \
                ['id',          'click',            'hour',         'weekday',          'C1',           'banner_pos',
                 'site_id',     'site_domain',      'site_category','app_id',           'app_domain',   'app_category',
                 'device_id',   'device_ip',        'device_model', 'device_type',      'device_conn_type', 'C14',
                 'C15',         'C16',              'C17',           'C18',             'C19',          'C20',
                 'C21',          'site_prob',        'app_prob',     'id_prob',          'ip_prob']

_FEATURE_MAP = {
    column_name: tf.FixedLenFeature(1, dtype=tf.int64) for column_name in COLUMN_NAMES[1:]
}

_COLUMN_DIM = \
                [0,             2,                  24,             7,                 7,                 7,
                 1101,          930,          26,                 895,           68,            36,
                 6288,          30805,       2429,               5,              4,              1390,
                 8,             9,             366,                4,              62,             172,
                 55,            21,             21,                 21,            21]

_COLUMN_EMBEDDING_DIM = \
                [0,             2,                  1,             1,              1,                      1,
                 1,           1,                 1,           1,             1,                     1,
                 1,           1,                1,              1,              1,                    1,
                 1,             1,                1,              1,             1,                    1,
                 1]

_COLUMN_EMBEDDING_DIM_FM = 10

_CSV_COLUMN_DEFAULTS = [[0]]*len(_COLUMN_DIM)
_CSV_COLUMN_DEFAULTS[0] = ['']


def build_model_columns():
    ctr_columns = [tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(feature_name, _COLUMN_DIM[index]),
        _COLUMN_EMBEDDING_DIM[index], max_norm=np.sqrt(_COLUMN_EMBEDDING_DIM[index]))
        for index, feature_name in enumerate(COLUMN_NAMES) if index > 1]
    return ctr_columns


def build_model_columns_fm():
    return dict(zip(COLUMN_NAMES[2:], _COLUMN_DIM[2:]))


def _deserialize(examples_serialized):
    features = tf.parse_example(examples_serialized, _FEATURE_MAP)
    label = features.pop('click')
    return features, label


def parse_csv(value):
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(COLUMN_NAMES, columns))
    features.pop('id')
    labels = features.pop('click')
    return features, labels


def get_input_fn(train_path, batch_size, repeat, shuffle, use_tfrecord=False):
    def csv_input_fn():
        # Extract lines from input files using the Dataset API.
        dataset = tf.data.TextLineDataset(train_path)
        #dataset = dataset.skip(1)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle)

        # We call repeat after shuffling, rather than before, to prevent separate
        # epochs from blending together.
        if repeat > 1:
            dataset = dataset.repeat(repeat)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(parse_csv, num_parallel_calls=8)
        return dataset.prefetch(batch_size*2)

    def tfrecord_input_fn():
        dataset = tf.data.TFRecordDataset(train_path)
        # batch comes before map because map can deserialize multiple examples.
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(_deserialize)
        if shuffle:
            dataset = dataset.shuffle(shuffle)

        dataset = dataset.repeat(repeat)
        return dataset.prefetch(2)

    if use_tfrecord:
        return tfrecord_input_fn
    else:
        return csv_input_fn
