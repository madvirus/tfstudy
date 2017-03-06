import tensorflow as tf

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable([[-0.69001538, 0.07054905, -1.43649220, -0.09584059, 3.13020134]])

h = tf.matmul(W, X)
hypothesis = tf.sigmoid(h)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

import numpy as np

xy = np.loadtxt('logistic_train2.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

guess = sess.run(hypothesis, feed_dict={X: x_data})
guess_y = np.transpose(np.vstack((guess[0], [y_data])))

for gy in guess_y:
    print(gy[0] , '\t',  gy[1])
