import numpy as np
import tensorflow as tf

xy = np.loadtxt('softmax_train.txt', unpack=False, dtype='float32')
x_data = xy[:, :2]  # [8, 2]
y_data = xy[:, 2:]  # [8, 3]

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 3])

W = tf.Variable(tf.zeros([2, 3]))
b = tf.Variable(tf.zeros([3]))

h = tf.nn.softmax(tf.matmul(X, W) + b)

a = 0.001

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(h), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(a).minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):
	sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
	if step % 200 == 0:
		print (step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))

a = sess.run(h, feed_dict={X: [[11, 7]]})
print ("a :", a, sess.run(tf.arg_max(a, 1)))

b = sess.run(h, feed_dict={X: [[3, 4]]})
print ("b :", b, sess.run(tf.arg_max(b, 1)))

c = sess.run(h, feed_dict={X: [[1, 0]]})
print ("c :", c, sess.run(tf.arg_max(c, 1)))
