from keras.models import load_model
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()


model = load_model('mlp.h5')

model.summary()

X_test = X_test.reshape(10000, 784)
X_test = X_test.astype('float32')
X_test /= 255

pc = model.predict_classes(X_test[0:100, :], 100)
print(pc)

print(pc == y_test[0:100])