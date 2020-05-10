import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten, Lambda
from keras import backend as K
import tensorflow as tf


from PIL import Image

import pickle

def get_data(filename):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)

    return dataset

def get_siamese_model(input_shape):
    """
        Model architecture
    """
    
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7,7), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,4), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,4), activation='relu'))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid'))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid')(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net

def get_conv_model(input_shape):
    """
        Model architecture
    """
  
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7,7), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,4), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,4), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2 ,activation='softmax'))

    return model


if __name__ == "__main__":
    # import os
    # Oos.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""


    print("Despickleando...")
    train = get_data("train_app1.p")
    val = get_data("val_app1.p")
    test = get_data("test_app1.p")
    print("Despickleado")

    X_train = train["X"]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    Y_train = train["Y"].T

    X_val = val["X"]
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2], 1))
    Y_val = val["Y"].T

    X_test = test["X"]
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    Y_test = test["Y"].T
    
    model = get_conv_model((X_train.shape[1], X_train.shape[2], 1))
    model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    epochs = 2
    batch_size = 64
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)