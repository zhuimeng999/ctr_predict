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

import os, re

import tensorflow as tf
from prev.ctr_dataset import build_model_columns, get_input_fn, FLAGS


tf.app.flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")


def model_tanh(net, scope, units, trainable):
    with tf.variable_scope(scope, default_name='tanh'):
        net1 = tf.layers.dense(net, units=units, activation=tf.nn.tanh, trainable=trainable)
        net2 = tf.layers.dense(net1, units=units, activation=tf.nn.tanh, trainable=trainable)
        logists = tf.layers.dense(net2, units=1, activation=None, trainable=trainable)

    return net1, net2, logists


def model_selu(net, scope, units, trainable):
    with tf.variable_scope(scope, default_name='selu'):
        net1 = tf.layers.dense(net, units=units, activation=tf.nn.selu, trainable=trainable)
        net2 = tf.layers.dense(net1, units=units, activation=tf.nn.selu, trainable=trainable)
        logists = tf.layers.dense(net2, units=1, activation=None, trainable=trainable)

    return net1, net2, logists


def model_relu(net, scope, units, trainable):
    with tf.variable_scope(scope, default_name='relu'):
        net1 = tf.layers.dense(net, units=units, activation=tf.nn.relu, trainable=trainable)
        net2 = tf.layers.dense(net1, units=units, activation=tf.nn.relu, trainable=trainable)
        logists = tf.layers.dense(net2, units=1, activation=None, trainable=trainable)

    return net1, net2, logists


model_factories = [model_tanh, model_selu, model_relu]


def my_model(features=None, labels=None, mode=None, config=None, params=None):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    input_dim = net.shape.as_list()
    tf.logging.info('input dim {}'.format(input_dim))

    assert 'models' in params and isinstance(params['models'], (list, tuple)) and len(params['models']) > 0

    models = []
    epoch_value = params['epoch_value']
    process_state = epoch_value // 5
    for i in range(process_state):
        models.append(params['models'][i](net, 'model' + str(i), input_dim[1], False))
    new_model_scope = 'model' + str(len(models))
    models.append(params['models'][process_state](net, new_model_scope, input_dim[1], True))
    #print(models[0][1].kernel)
    logists = sum(map(lambda x: x[-1], models))
    # Compute predictions.
    predicted_classes = tf.cast(tf.greater(logists, 0), tf.int64)
    probabilities = tf.nn.sigmoid(logists)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': probabilities
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    # Compute loss.
    loss = tf.losses.log_loss(labels=labels, predictions=probabilities)

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

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    basic_list = []
    extand_list = []
    for tensor_var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if re.match('.*' + new_model_scope + '/.*', tensor_var.name):
            extand_list.append(tensor_var)
        else:
            basic_list.append(tensor_var)

    class MySaver(tf.train.Saver):
        def __init__(self, basic_list=None, extand_list=None):
            if basic_list is None:
                self._basic_saver = None
                full_list = extand_list
            else:
                self._basic_saver = tf.train.Saver(var_list=basic_list)
                full_list = [*basic_list, *extand_list]
            super().__init__(var_list=full_list)
            self._basic_list = basic_list
            self._extand_list = extand_list
            self._is_restored = False

        def restore(self, sess, save_path):
            try:
                super().restore(sess, save_path)
            except tf.errors.NotFoundError as e:
                assert not self._is_restored

                sess.run([v.initializer for v in self._extand_list])
                if self._basic_saver:
                    self._basic_saver.restore(sess, save_path)
            self._is_restored = True

    scaffold = tf.train.Scaffold(saver=MySaver(basic_list=basic_list, extand_list=extand_list))
    hook = tf.train.CheckpointSaverHook(
        config.model_dir,
        save_secs=config.save_checkpoints_secs,
        save_steps=config.save_checkpoints_steps,
        scaffold=scaffold)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, scaffold=scaffold, training_hooks=[hook])


class boost(tf.estimator.Estimator):
    def __init__(self, model_fn, model_dir=None, config=None, params=None,
                 warm_start_from=None):
        super().__init__(model_fn=model_fn, model_dir=model_dir, config=config, params=params, warm_start_from=warm_start_from)

        def _train(input_fn, hooks=None, steps=None, max_steps=None):
            ckpt = tf.train.get_checkpoint_state(self._model_dir)
            if ckpt:
                print(ckpt.model_checkpoint_path)
                reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
                self._params['epoch_value'] = reader.get_tensor('epoch_tensor')
                print('############epoch_value', self._params['epoch_value'])
            else:
                self._params['epoch_value'] = 0
            top_self = self

            class MyHook(tf.train.SessionRunHook):
                def end(self, session):
                    print(session.run([top_self._epoch_add_tensor]))
                    return super().end(session)

            saver_hook = MyHook()
            if hooks:
                hooks.append(saver_hook)
            else:
                hooks = [saver_hook]
            result = super(boost, self).train(input_fn=input_fn, hooks=hooks, steps=steps, max_steps=max_steps)

            return result
        self.train = _train

    def _call_model_fn(self, features, labels, mode, config):
        epoch_tensor = tf.get_variable('epoch_tensor', shape=(), dtype=tf.int32, initializer=tf.zeros_initializer,trainable=False)
        epoch_add_tensor = tf.assign_add(epoch_tensor, 1)
        self._epoch_tensor = epoch_tensor
        self._epoch_add_tensor = epoch_add_tensor
        return super()._call_model_fn(features, labels, mode, config)


def main(_):
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    if True:
        ws = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from="/home/lucius/Projects/notebook/homework/ctr_predict/models/v2",
            vars_to_warm_start=".*input_layer/.*")
    else:
        ws = None
    classifier = boost(
        model_fn=my_model,
        params={
            'feature_columns': build_model_columns(),
            'models': model_factories
        },
        model_dir=FLAGS.output_dir,
        warm_start_from=ws)
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
