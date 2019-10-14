# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.datasets import mnist
from keras.utils import np_utils
import os
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tkinter
 
def read():

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # load the dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # let's print the shape before we reshape and normalize
    # print("\nX_test shape", X_test.shape)
    # print("\ny_test shape", y_test.shape)

    # building the input vector from the 28x28 pixels
    X_test = X_test.reshape(10000, 784)
    X_test = X_test.astype('float32')

    # normalizing the data to help with the training
    X_test /= 255

    # print the final input shape ready for training
    # print("\nTest matrix shape", X_test.shape)

    # one-hot encoding using keras' numpy-related utilities
    n_classes = 10
    classes=[0,1,2,3,4,5,6,7,8,9]
    Y_test = np_utils.to_categorical(y_test, n_classes)
    # print("\nShape after one-hot encoding: ", Y_test.shape)
    
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy','binary_accuracy',
                            'categorical_accuracy'])
    score = loaded_model.evaluate(X_test, Y_test, verbose=0)

    print("\n ---> Metricas del Sistema")
    print("\n%s: %.2f%%" % (loaded_model.metrics_names[0], score[0]*100))
    print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    print("\n%s: %.2f%%" % (loaded_model.metrics_names[2], score[2]*100))
    print("\n%s: %.2f%%" % (loaded_model.metrics_names[3], score[3]*100))
    
    y_pred = loaded_model.predict(X_test)
    con_mat = tf.math.confusion_matrix(labels=Y_test.argmax(axis=1), predictions=y_pred.argmax(axis=1)).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index = classes, columns = classes)

    print("\n ---> Matriz de COnfusión del Sistema\n")
    print(con_mat_df)
    print()

    return loaded_model