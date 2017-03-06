from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.regularizers import l2
from mnist import load_mnist

(X_train, Y_train), (X_test, Y_test) = load_mnist(
    normalize=True, one_hot_label=True
)

model = Sequential()
model.add(Dense(100, input_shape=(784,), init='he_normal', W_regularizer=l2(0.01)))
model.add(Activation("sigmoid"))
model.add(Dense(10))
model.add(Activation("softmax"))

model.summary()

sgd = SGD(lr=0.1)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

batch_size = 100
nb_epoch = 1  # 1 epoch = 100 * 600
his = model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          verbose=2, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test)
print(model.metrics_names)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#model.save('mlp.h5')
