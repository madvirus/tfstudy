# -*- coding: utf-8 -*-

x_data = [1.9, 2.5, 3.2, 3.8, 4.7, 5.5, 5.9, 7.2]
y_data = [22, 33, 30, 42, 38, 49, 42, 55]

import tensorflow as tf

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.015)  # 0.1로 테스트하면 커지는 증상 확인 가능
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print('FIRST', sess.run(loss), sess.run(W), sess.run(b))

for step in range(1800):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(W), sess.run(b))

import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['font.family'] = 'NanumBarunGothic'

import matplotlib.pyplot as plt

plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.xlabel('햇볕')
plt.ylabel('관객수')
plt.legend()
plt.show()

