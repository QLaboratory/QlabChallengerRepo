from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def ensemble_net(features, labels, mode):

    """Model function for Net."""

    # 输入层，输入张量为 [batch_size, width, channels]
    channels = 5 #要和网络数一致
    input_layer = tf.reshape(features["x"], [-1, 80, channels])

    # SE模块
    se = tf.reduce_mean(input_layer, 1, name='global_avg_pooling')
    se = tf.layers.dense(inputs=se, units=channels, activation=tf.nn.relu, name='fc1')
    se = tf.layers.dense(inputs=se, units=channels, activation=tf.nn.sigmoid, name='fc2')

    l = input_layer * tf.reshape(se, [-1, 1, channels])
    l = tf.layers.conv1d(inputs = l, filters=1, strides=1, kernel_size=1)
    
    # 去除多余维数
    l = tf.squeeze(l)
    # 预测层
    predictions = {
        "classes": tf.argmax(input=l, axis=1),
        "probabilities": tf.nn.softmax(l, name="softmax_tensor")
        }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # 读入测试集标签
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=80)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=l)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # 精度评估
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    # 加载训练数据
    # TODO json格式读取
    train_data = np.arange(1000*80*5, dtype=np.float32).reshape(1000, 80, 5)  # 5要和网络数一致
    train_labels = np.arange(1000, dtype=np.int32).reshape(1000)

    eval_data = np.arange(200*80*5, dtype=np.float32).reshape(200, 80, 5)  # 5要和网络数一致
    eval_labels = np.arange(200, dtype=np.int32).reshape(200)

    ensemble_net_classifier = tf.estimator.Estimator(
        model_fn=ensemble_net, model_dir="/tmp/ensemble_net")    

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=5)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=10,
        num_epochs=None,
        shuffle=True)
    ensemble_net_classifier.train(input_fn=train_input_fn, steps=100, hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = ensemble_net_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
  
if __name__ == "__main__":
    tf.app.run()