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
"""Train DNN on Kaggle movie dataset."""

import os
import tensorflow as tf
from models.v1.v1_dataset import build_model_columns_fm, get_input_fn, COLUMN_NAMES

def get_my_input(dataset):
    iterator = dataset.make_one_shot_iterator()

    # Create saveable object from iterator.
    saveable = tf.contrib.data.make_saveable_from_iterator(iterator)

    # Save the iterator state by adding it to the saveable objects collection.
    tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)


    next_element = iterator.get_next()


def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.

    #lr
    regularizer = None
    with tf.variable_scope('input_layer'):
        lr_list = []
        for k, v in features.items():
            embedding_table = tf.get_variable(k + '_embed_lr', [params['feature_columns'][k], 1], regularizer=regularizer)
            embedded_var = tf.nn.embedding_lookup(embedding_table, v)
            lr_list.append(embedded_var)

        fm_list = []
        for k, v in features.items():
            embedding_table = tf.get_variable(k + '_embed_fm', [params['feature_columns'][k], 18], regularizer=regularizer)
            embedded_var = tf.nn.embedding_lookup(embedding_table, v)
            fm_list.append(embedded_var)

        lr_bias = tf.get_variable('lr_bias', shape=(), dtype=tf.float32, initializer=tf.truncated_normal_initializer,
                                  regularizer=regularizer)

    fm_matrix = tf.stack(fm_list, axis=2)
    sum_squre = tf.reduce_sum(fm_matrix, axis=2)
    sum_squre = tf.square(sum_squre)
    squre_sum = tf.square(fm_matrix)
    squre_sum = tf.reduce_sum(squre_sum, axis=2)
    lr_list.append(squre_sum)
    lr_list.append(-sum_squre)

    output = tf.concat(lr_list, axis=2)
    logits = tf.reduce_sum(output, axis=2) + lr_bias
    # Compute predictions.
    predicted_classes = tf.cast(tf.greater(logits, 0), tf.int64)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.sigmoid(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    # Compute loss.
    loss = tf.losses.log_loss(labels=labels, predictions=tf.nn.sigmoid(logits), reduction=tf.losses.Reduction.MEAN)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    tf.summary.scalar('train_loss', loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    loss = loss + tf.losses.get_regularization_loss()
    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    adam_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    exp_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=tf.train.get_global_step(),
                                          decay_rate=0.8, decay_steps=20000)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=exp_rate)
    sgd_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if ('optimizer' in params) and (params['optimizer'] == 'sgd'):
        train_op = sgd_op
    else:
        train_op = adam_op

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(_):
    model_dir = os.path.join(os.path.dirname(__file__), 'logdata_fm_noreg')
    restore_map = {'input_layer/' + feature_name + '_embed_lr':feature_name + '_embed_lr' for feature_name in COLUMN_NAMES[2:]}
    restore_map.update({'input_layer/' + feature_name + '_embed_fm': feature_name + '_embed_lr' for feature_name in
                   COLUMN_NAMES[2:]})
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    if True:
        ws = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from="/home/lucius/Projects/notebook/homework/ctr_predict/models/v2/logdata_3_4",
            vars_to_warm_start=".*input_layer.*")
    else:
        ws = None

    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': build_model_columns_fm(),
            'optimizer': 'sgd'
        },
        model_dir=model_dir,
        warm_start_from=ws)

    for i in range(15):
        print('perform {}s epoch...'.format(i))
        # Train the Model.
        classifier.train(
            input_fn=get_input_fn(os.path.join(os.path.dirname(__file__), '../../FE/FE1/train_split.tfrecord'), 256, 1, 5000, use_tfrecord=True))

        # Evaluate the model.
        eval_result = classifier.evaluate(
            input_fn=get_input_fn(os.path.join(os.path.dirname(__file__), '../../FE/FE1/valid_split.tfrecord'), 256, 1, 5000, use_tfrecord=True))

        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    # expected = ['Setosa', 'Versicolor', 'Virginica']
    # predict_x = {
    #     'SepalLength': [5.1, 5.9, 6.9],
    #     'SepalWidth': [3.3, 3.0, 3.1],
    #     'PetalLength': [1.7, 4.2, 5.4],
    #     'PetalWidth': [0.5, 1.5, 2.1],
    # }
    #
    # predictions = classifier.predict(
    #     input_fn=get_input_fn(os.path.join('output', 'train_format.tfrecord'), 256, 1, 1000))
    #
    # for pred_dict, expec in zip(predictions, expected):
    #     template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
    #
    #     class_id = pred_dict['class_ids'][0]
    #     probability = pred_dict['probabilities'][class_id]
    #
    #     print(template.format(iris_data.SPECIES[class_id],
    #                           100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
