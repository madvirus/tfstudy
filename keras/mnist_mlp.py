'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

batch_size = 256
nb_classes = 10
nb_epoch = 1

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("train shape: ", X_train.shape, y_train.shape)  # x_train (60000, 28, 28), y_train (60000,)
print("train x value: ", X_train[0])  # x_train: 흑백이미지 (한 픽셀당 0~255)
print("train y label: ", y_train[0])  # label 값이 0부터 9사이 값
print("test shape: ", X_test.shape, y_test.shape)  # x_train (60000, 28, 28), y_train (60000,)
print("test y label: ", y_test[0])  # label 값이 0부터 9사이

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print(X_test[0:1, :].shape, 'first test sample')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)  # Y_train : (60000, 10)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print("y_train:", y_train.shape, "  -->   Y_train: ", Y_train.shape, Y_train[0])

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save('mlp.h5')

pc = model.predict(X_test[0:1, :], 1)
print(pc, 'predict')
pc = model.predict_classes(X_test[0:1, :], 1)
print(pc, 'predict_classes')
