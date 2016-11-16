import numpy as np

x_data = np.array([1.9, 2.5, 3.2, 3.8, 4.7, 5.5, 5.9, 7.2])
y_data = np.array([22, 33, 30, 42, 38, 49, 42, 55])

wrange = [i/10 - 9.7 for i in range(300)]

loss = [np.mean(np.square(np.array(x_data * w + 15.7) - y_data)) for w in wrange]

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

plt.plot(wrange, loss)
plt.xlabel('w')
plt.ylabel('loss')
plt.legend()
plt.show()
