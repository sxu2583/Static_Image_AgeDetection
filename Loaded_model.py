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
from Age_Detection import parse_Dataset
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main():
    # Reload a saved model(Architecutre and weights) and run the prediction
    number_of_epochs = int(input("How many Epochs: "))
    x_train, y_train, x_test, y_test, size = parse_Dataset()

    new_model = load_model('models/100_epochs.h5')
    #new_model.fit(x_train, y_train, validation_data=(x_test, y_test),
    # epochs=number_of_epochs)

main()