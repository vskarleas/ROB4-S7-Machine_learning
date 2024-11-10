
# Import libraries and modules
import numpy as np
import time
np.random.seed(123)  # for reproducibility

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import  Dense, Flatten, Dropout, Convolution2D, MaxPooling2D
from utilitaire import affiche





##################################################
# I - Load pre-shuffled MNIST data train and test sets
##################################################
from tensorflow.keras.datasets.mnist import load_data
# load dataset
(X_train, y_train), (X_test, y_test) = load_data()

#Ne conserve que 10% de la base
X_train, pipo, y_train, pipo = train_test_split(X_train, y_train, test_size=0.95)
X_test, pipo, y_test, pipo = train_test_split(X_test, y_test, test_size=0.95)


for i in range(200):
  plt.subplot(10,20,i+1)
  plt.imshow(X_train[i,:].reshape([28,28]), cmap='gray')
  plt.axis('off')
plt.show()

# Preprocess input data
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


Y_train = tf.keras.utils.to_categorical(y_train, 10)
Y_test = tf.keras.utils.to_categorical(y_test, 10)


##################################################
# II - Régression logistiques
##################################################
# II.1. Define model architecture
# Define model architecture
inputs = Input(shape=(28,28,1))
x = inputs
x=Flatten()(x)
outputs=Dense(10, activation='softmax')(x)
model = Model(inputs, outputs)
model.summary()

# II.2. Apprentissage
lr= ??
batch_size= 256
epochs=10

sgd1= tf.keras.optimizers.SGD(learning_rate=lr)

model.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=['accuracy'])
tps1 = time.time()


history =model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1,validation_data=(X_test, Y_test))
tps2 = time.time()
#print(history.history.keys())

affiche(history)
print('lr=',lr,'batch_size=',batch_size, 'epochs=',epochs)
print('Temps d apprentissage',tps2 - tps1)

# II.3. Evaluation du modèle
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=-1)
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))

