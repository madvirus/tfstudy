# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

x = tf.placeholder("float", [None, 784], "x") # 입력 이미지 (n개 * 784픽셀)
W = tf.Variable(tf.zeros([784, 10]), name="W") # 784 각 픽셀에 대해 각 라벨에 대한 가중치
b = tf.Variable(tf.zeros([10]), name="b") # 각 라벨에 대한 bias

matm = tf.matmul(x, W)
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 10], name="y_") # 트레이닝 데이터의 라벨

cross_entropy = -tf.reduce_sum(y_ * tf.log(y)) # 비용 함수

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) # cross_entropy를 최소화하는 방향으로

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
