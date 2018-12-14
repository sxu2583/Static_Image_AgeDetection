from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
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

    # Size overall = 52938
    x_train = np.array(data_images[0:50000])
    y_train = np.array(data_ages[0:50000])

    x_test = np.array(data_images[50000:])
    y_test = np.array(data_ages[50000:])

    # The number of images, size of images, grayscale indicator
    x_train = x_train.reshape(50000, 28, 28, 1)
    x_test = x_test.reshape(2938, 28, 28, 1)

    # One-hot encode the ages/results
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


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

def run_model(x_train, y_train, x_test, y_test, plot_type, epochs):
    # Create the sequential model from keras (CNN)
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    
    # Compile the model - include the accuracy metric and loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Training the model
    number_of_epochs = epochs
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=number_of_epochs, verbose=1)

    model.save('models/' + str(number_of_epochs) +'_epochs.h5')


def main():
    epochs = int(input("How many epochs: "))
    plot = PlotLosses()
    x_train, y_train, x_test, y_test = parse_Dataset()
    run_model(x_train, y_train, x_test, y_test, plot, epochs)
    print("Model Saved")
main()
