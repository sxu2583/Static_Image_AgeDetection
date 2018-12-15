from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import math
import os
import keras
import datetime
from IPython.display import clear_output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def parse_Dataset():
    # Overall dataset
    data_images = []
    data_ages = []

    # Read in the images by id and store dataframe into x_train and age into y_train
    for f in range(100):
        with open('Json/' + str(f) + '.json') as data_file:
            data = json.load(data_file)
            for v in data.values():
                data_images.append(v['img'])
                age_bin = math.ceil((v['age'] / 10)) - 1
                data_ages.append(age_bin)

    # Check the shape of the dataframes
    data_images = np.array(data_images)
    data_ages = np.array(data_ages)

    size = len(data_images[0])

    # Size overall = 52938
    x_train = np.array(data_images[0:50000])
    y_train = np.array(data_ages[0:50000])

    x_test = np.array(data_images[50000:])
    y_test = np.array(data_ages[50000:])

    # The number of images, size of images, grayscale indicator
    x_train = x_train.reshape(50000, size, size, 1)
    x_test = x_test.reshape(2939, size, size, 1)

    # One-hot encode the ages/results
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test, size


# Code snippet from gitlab for plotting (Just for testing)
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()


def run_model(x_train, y_train, x_test, y_test, plot_type, epochs, size):
    # Create the sequential model from keras (CNN)
    model = Sequential()

    # Add the layers to the sequential model
    model.add(Conv2D(64, kernel_size=3, activation='relu',
                     input_shape=(size, size, 1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Training the model
    number_of_epochs = epochs
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=number_of_epochs, verbose=1)
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    history.save('models/' + str(number_of_epochs) + '_epochs.h5')


def main():
    epochs = int(input("How many epochs: "))
    plot = PlotLosses()
    x_train, y_train, x_test, y_test, size = parse_Dataset()
    run_model(x_train, y_train, x_test, y_test, plot, epochs, size)
    print("Model Saved")
main()
