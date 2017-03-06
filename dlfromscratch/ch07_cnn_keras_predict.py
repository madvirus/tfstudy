from keras.models import load_model
from mnist import load_mnist
import numpy as np

(X_train, Y_train), (X_test, Y_test) = load_mnist(
    normalize=True, one_hot_label=True
)

img_rows, img_cols = 28, 28
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

model = load_model('cnn.h5')

pp = model.predict_proba(X_test[0:100, :, :, :], 10)
pc = model.predict_classes(X_test[0:100, :], 10)

print("predict_proba")
print(pp)
print("predict_classes")
print(pc)
print("np.argmax")
print(np.argmax(Y_test[0:100], axis=1))
print("true:false")
print(pc == np.argmax(Y_test[0:100], axis=1))
