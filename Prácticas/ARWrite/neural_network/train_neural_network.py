"""
Script that trains a neural network to predict letters by using MNIST dataset.
"""
import os

import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
from mnist import MNIST
from sklearn.model_selection import train_test_split
from tensorflow.python.util import deprecation as deprecation

import parameters

# Do not show deprecation warnings
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Select MNIST letters dataset and load it
emnist_data = MNIST(path=parameters.DATA_PATH, return_type='numpy')
emnist_data.select_emnist('letters')
# Load training data
x, y = emnist_data.load_training()
x = x.reshape(124800, 28, 28)
y = y.reshape(124800, 1)
# To deal with 0 index
y -= 1

# Split the dataset into train (75%) and test (25%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=111)
x_train = x_train.reshape(x_train.shape[0], parameters.IMG_WIDTH, parameters.IMG_HEIGHT, 1)
x_test = x_test.reshape(x_test.shape[0], parameters.IMG_WIDTH, parameters.IMG_HEIGHT, 1)

input_shape = (parameters.IMG_WIDTH, parameters.IMG_HEIGHT, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Change image values to 0/1
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, parameters.NUMBER_OF_LETTERS)
y_test = keras.utils.to_categorical(y_test, parameters.NUMBER_OF_LETTERS)

# Build the neural network  Conv->Conv->MaxPooling->Dropout->Flatten->Dense->Droput->Dense
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(parameters.NUMBER_OF_LETTERS, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=parameters.BATCH_SIZE,
          epochs=parameters.NUMBER_OF_EPOCHS, verbose=1, validation_data=(x_test, y_test))

# Save the model into a file to use it in the main app.
model.save(parameters.NEURAL_NETWORK_FILE)

# Load the model to check that everything is OK.
model = load_model(parameters.NEURAL_NETWORK_FILE)
# Evaluate it with the test dataset.
score = model.evaluate(x_test, y_test, verbose=0)
print('Loss:', score[0])
print('Accuracy:', score[1])
