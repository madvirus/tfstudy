# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

if tf.gfile.Exists("nnlog"):
    tf.gfile.DeleteRecursively("nnlog")
tf.gfile.MakeDirs("nnlog")

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

x = tf.placeholder("float", [None, 784], "x") # 입력 이미지 (n개 * 784픽셀)

y_ = tf.placeholder("float", [None, 10], name="y_") # 트레이닝 데이터의 라벨

with tf.name_scope('layer1'):
    with tf.name_scope('W'):
        W = tf.Variable(tf.zeros([784, 10]), name="W") # 784 각 픽셀에 대해 각 라벨에 대한 가중치
        variable_summaries(W)
    with tf.name_scope('b'):
        b = tf.Variable(tf.zeros([10]), name="b") # 각 라벨에 대한 bias
        variable_summaries(b)

    matm = tf.matmul(x, W)
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    tf.summary.histogram('y', y)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y)) # 비용 함수
tf.name_scope('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) # cross_entropy를 최소화하는 방향으로

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('accuracy', accuracy)

sess = tf.InteractiveSession()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("nnlog/train", sess.graph)
test_writer = tf.summary.FileWriter("nnlog/test")

sess.run(tf.global_variables_initializer())

for i in range(200):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i % 10 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
    else:
        if i % 20 == 19:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys},
                                  options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            print('Adding run metadata for', i)
        else:
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
            train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()
