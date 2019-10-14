# imports for array-handling and plotting
import numpy as np
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns

# let's keep our keras backend tensorflow quiet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

def create():

    # load the dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # let's print the shape before we reshape and normalize
    # print("\nX_train shape", X_train.shape)
    # print("\ny_train shape", y_train.shape)
    # print("\nX_test shape", X_test.shape)
    # print("\ny_test shape", y_test.shape)

    # building the input vector from the 28x28 pixels
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalizing the data to help with the training
    X_train /= 255
    X_test /= 255

    # print the final input shape ready for training
    # print("\nTrain matrix shape", X_train.shape)
    # print("\nTest matrix shape", X_test.shape)

    # print(np.unique(y_train, return_counts=True))

    # one-hot encoding using keras' numpy-related utilities
    n_classes = 10
    classes=[0,1,2,3,4,5,6,7,8,9]
    # print("\nShape before one-hot encoding: ", y_train.shape)
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)
    # print("\nShape after one-hot encoding: \n", Y_train.shape)

    # building a linear stack of layers with the sequential model
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))                            
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy','binary_accuracy',
                            'categorical_accuracy'], optimizer='adam')

    # training the model and saving metrics in history
    history = model.fit(X_train, Y_train,
            batch_size=128, epochs=20,
            verbose=2,
            validation_data=(X_test, Y_test))

    # evaluate the model
    scores = model.evaluate(X_train, Y_train, verbose=0)

    print("\n ---> Metricas del Sistema")
    print("\n%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("\n%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    print("\n%s: %.2f%%" % (model.metrics_names[3], scores[3]*100))

    y_pred = model.predict(X_test)
    con_mat = tf.math.confusion_matrix(labels=Y_test.argmax(axis=1), predictions=y_pred.argmax(axis=1)).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index = classes, columns = classes)

    print("\n ---> Matriz de Confusión del Sistema\n")
    print(con_mat_df)
    print()

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    return model



def create_graphics():

    # load the dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # let's print the shape before we reshape and normalize
    # print("\nX_train shape", X_train.shape)
    # print("\ny_train shape", y_train.shape)
    # print("\nX_test shape", X_test.shape)
    # print("\ny_test shape", y_test.shape)

    # building the input vector from the 28x28 pixels
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalizing the data to help with the training
    X_train /= 255
    X_test /= 255

    # print the final input shape ready for training
    # print("\nTrain matrix shape", X_train.shape)
    # print("\nTest matrix shape", X_test.shape)

    # print(np.unique(y_train, return_counts=True))

    # one-hot encoding using keras' numpy-related utilities
    n_classes = 10
    classes=[0,1,2,3,4,5,6,7,8,9]
    # print("\nShape before one-hot encoding: ", y_train.shape)
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)
    # print("\nShape after one-hot encoding: \n", Y_train.shape)

    # building a linear stack of layers with the sequential model
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))                            
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy','binary_accuracy',
                            'categorical_accuracy'], optimizer='adam')

    # training the model and saving metrics in history
    history = model.fit(X_train, Y_train,
            batch_size=128, epochs=20,
            verbose=2,
            validation_data=(X_test, Y_test))

    # evaluate the model
    scores = model.evaluate(X_train, Y_train, verbose=0)

    print("\n ---> Metricas del Sistema")
    print("\n%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("\n%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    print("\n%s: %.2f%%" % (model.metrics_names[3], scores[3]*100))

    y_pred = model.predict(X_test)
    con_mat = tf.math.confusion_matrix(labels=Y_test.argmax(axis=1), predictions=y_pred.argmax(axis=1)).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index = classes, columns = classes)

    print("\n ---> Matriz de Confusión del Sistema\n")
    print(con_mat_df)
    print()

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    # Graphics of loss, acurracy and confusion matrix

    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

    plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], history.history['loss'], 'k')
    plt.title('Loss function decrease', fontdict=font)
    plt.text(2, 0.65, r'LOSS', fontdict=font)
    plt.xlabel('Batch Steps', fontdict=font)
    plt.ylabel('Loss funct', fontdict=font)

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()

    plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], history.history['accuracy'], 'k')
    plt.title('Acurracy update', fontdict=font)
    plt.text(2, 0.65, r'ACCURACY', fontdict=font)
    plt.xlabel('Batch Steps', fontdict=font)
    plt.ylabel('Accracy', fontdict=font)

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return model

# create_graphics()