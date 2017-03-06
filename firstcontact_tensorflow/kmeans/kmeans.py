# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('TkAgg')

num_points = 2000
vectors_set = []

for i in range(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.0, 0.9),
                            np.random.normal(0.0, 0.9)])
    else:
        vectors_set.append([np.random.normal(3.0, 0.5),
                            np.random.normal(1.0, 0.5)])


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.DataFrame({"x": [v[0] for v in vectors_set], "y": [v[1] for v in vectors_set] })

sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
plt.show()

import tensorflow as tf

vectors = tf.constant(vectors_set) # 2차 텐서 (2000, 2)
k = 4

shuffled_vectors = tf.random_shuffle(vectors) # (2000, 2)
slided_vectors = tf.slice(shuffled_vectors, [0, 0], [k, -1]) # (4, 2)
centroides = tf.Variable(slided_vectors) # (4, 2)

expanded_vectors = tf.expand_dims(vectors, 0) # (1, 2000, 2)
expanded_centroides = tf.expand_dims(centroides, 1) # (4, 1, 2)

diff_result = tf.sub(expanded_vectors, expanded_centroides) # (4, 2000, 2) - 각 요소의 값 차이, 즉 2점 좌표의 값 차이
square_result = tf.square(diff_result) # (4, 2000, 2)
sum_result = tf.reduce_sum(square_result, 2) # (4, 2000)
assignments = tf.argmin(sum_result, 0) # (2000), D0 차원 기준으로 D1 차원 값이 가장 작은 것 선택

means = tf.concat(0,  # D0 차원에 추가
                  [
                      tf.reduce_mean(
                          tf.gather(
                              vectors,
                              tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])
                          ),
                          reduction_indices=[1]
                      )
                      for c in range(k)
                  ]
                  )
update_centroides = tf.assign(centroides, means)

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

for step in range(50):
    _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])

data = {"x": [], "y": [], "cluster": []}

for i in range(len(assignment_values)):
    data["x"].append(vectors_set[i][0])
    data["y"].append(vectors_set[i][1])
    data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
plt.show()

