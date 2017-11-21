from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import json

SUMMARY_LOG_SAVE_PATH = ""

DENSENET_MODEL_PREDICT_RESULT_FILE = ""
RESNET_MODEL_PREDICT_RESULT_FILE = ""
XCEPTION_MODEL_PREDICT_RESULT_FILE = ""

def VariableSummaries(var):
  #记录张量
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def feed_dict(train, batch_size=256):
  # TODO 实现标签数据的选取
  if train:
    xs, ys = mnist.train.next_batch(batch_size)
  else:
    xs, ys = mnist.test.images, mnist.test.labels

  return {x: xs, y_: ys}


def main(unused_argv):

    #数据参数
    net_num = 3
    img_num = 50000
    class_num = 80
    epoch = 10
    batch_size = 256
    learning_rate = 0.001

    # 加载训练数据
    # TODO json格式读取 feed_dict函数实现
    train_data = np.arange(1000*80*5, dtype=np.float32).reshape(1000, 80, 5)  # 5要和网络数一致
    train_labels = np.arange(1000, dtype=np.int32).reshape(1000)

    eval_data = np.arange(200*80*5, dtype=np.float32).reshape(200, 80, 5)  # 5要和网络数一致
    eval_labels = np.arange(200, dtype=np.int32).reshape(200)

    sess = tf.InteractiveSession()

    #输入占位
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, class_num, net_num], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, class_num], name='y-input')

    '''SE模块'''
    #全局池化
    global_max_pooling = tf.reduce_max(x, 1, name='global_max_pooling')
    VariableSummaries(global_max_pooling)

    #第一层全连接
    with tf.name_scope('fc1'):
        fc1 = tf.layers.dense(inputs=global_max_pooling, units=net_num, activation=tf.nn.relu, name='fc1')
    VariableSummaries(fc1)

    #第二层全连接
    with tf.name_scope('fc2'):
        fc2 = tf.layers.dense(inputs=fc1, units=net_num, activation=tf.nn.sigmoid, name='fc2')
    VariableSummaries(fc2)

    #加权
    with tf.name_scope('scale'):
        scale = tf.reshape(fc2, [-1, 1, net_num])
        scaled_x = x * fc2
    VariableSummaries(scaled_x)

    with tf.name_scope('cov1'):
        sum_x = tf.layers.conv1d(inputs=scaled_x, filters=1, strides=1, kernel_size=1)
        # 去除多余维数
        y = tf.squeeze(sum_x)
    VariableSummaries(y)

    #计算交叉熵
    with tf.name_scope('cross_entropy'):
      diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
      with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    #训练步骤
    with tf.name_scope('train'):
      train_step = tf.train.AdamOptimizer(learning_rate).minimize(
          cross_entropy)

    #精度评估
    with tf.name_scope('accuracy'):
      with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    #tensorboard记录
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(SUMMARY_LOG_SAVE_PATH + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(SUMMARY_LOG_SAVE_PATH + '/test')
    tf.global_variables_initializer().run()

    #开始训练或预测
    for i in range(img_num//batch_size*epoch):
      if i % 20 == 0:  # Record summaries and test-set accuracy
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
      else:  # Record train set summaries, and train
        if i % 40 == 39:  # Record execution stats
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()
          summary, _ = sess.run([merged, train_step],
                                feed_dict=feed_dict(True, batch_size=batch_size),
                                options=run_options,
                                run_metadata=run_metadata)
          train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
          train_writer.add_summary(summary, i)
          print('Adding run metadata for', i)
        else:  # Record a summary
          summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True, batch_size=batch_size))
          train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()


if __name__ == "__main__":
    tf.app.run(main=main)
