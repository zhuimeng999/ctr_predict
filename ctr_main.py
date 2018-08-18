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
from ctr_dataset import build_model_columns, get_input_fn, FLAGS

tf.app.flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")


def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    input_dim = net.shape.as_list()
    tf.logging.info('input dim {}'.format(input_dim))
    net = tf.layers.dense(net, units=input_dim[1], activation=tf.nn.tanh)
    net = tf.layers.dense(net, units=input_dim[1], activation=tf.nn.tanh)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, units=1, activation=None)
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
    loss = tf.losses.log_loss(labels=labels, predictions=tf.nn.sigmoid(logits))

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(_):
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': build_model_columns()
        },
        model_dir=FLAGS.output_dir)

    for i in range(FLAGS.epochs_to_train):
        print('perform {}s epoch...'.format(i))
        # Train the Model.
        classifier.train(
            input_fn=get_input_fn(os.path.join('prev-output', 'train_format.tfrecord'), 256, 1, 1000, use_tfrecord=True))

        # Evaluate the model.
        eval_result = classifier.evaluate(
            input_fn=get_input_fn(os.path.join('prev-output', 'valid_format.tfrecord'), 256, 1, 1000, use_tfrecord=True))

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
