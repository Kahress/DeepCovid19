import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten, Lambda
from keras import backend as K
import tensorflow as tf
import random


from PIL import Image

import pickle

num_classes = 2

def get_data(filename):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)

    return dataset

def get_siamese_model(input_shape):
    """
        Model architecture
    """
    
    # Define the tensors for the two input images
    left_input = Input(input_shape, name = 'LeftInput')
    right_input = Input(input_shape, name = 'RigthInput')
    
    # Convolutional Neural Network
    model = Sequential(name='SiameseModel')
    model.add(Conv2D(64, (10,10), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7,7), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,4), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,4), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='sigmoid'))
    
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

def make_pairs(X, Y):
    classes = [X[Y==i] for i in range(num_classes)]
    len_classes = [c.shape[0] for c in classes]
    
    pairs = []
    labels = []

    n = min(len_classes)

    num_c = 0
    for c, num_samples in zip(classes, len_classes) :
        for i in range(n):
            z1, z2 = c[i % num_samples], c[(i + 1) % num_samples]
            pairs.append([z1, z2])
            inc = random.randrange(1, num_classes)
            icn = (num_c + inc) % num_classes
            z3, z4 = c[i % num_samples], classes[icn][i % len_classes[icn]]
            pairs.append([z3, z4])
            labels += [1, 0]
        num_c += 1
    return np.array(pairs), np.array(labels)

if __name__ == "__main__":
    print("Unpickling...")
    train = get_data("train_app1.p")
    val = get_data("val_app1.p")
    test = get_data("test_app1.p")
    print("Unpickled!")

    X_train = train["X"].astype('float32')
    X_train /= 255.0
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    Y_train = train["Y"].T
    y_train = train["y"]

    X_val = val["X"].astype('float32')
    X_val /= 255.0
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2], 1))
    Y_val = val["Y"].T
    y_val = val["y"]

    X_test = test["X"].astype('float32')
    X_test /= 255.0
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    Y_test = test["Y"].T
    y_test = test["y"]

    # model = get_conv_model((X_train.shape[1], X_train.shape[2], 1))
    model = get_siamese_model((X_train.shape[1], X_train.shape[2], 1))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    print("Making pairs...")
    X_train_pairs, y_train_pairs = make_pairs(X_train, y_train)
    
    X_val_pairs, y_val_pairs = make_pairs(X_val, y_val)
    print("Pairs made")

    epochs = 2
    batch_size = 5
    model.fit([X_train_pairs[:,0], X_train_pairs[:,1]], y_train_pairs, epochs=epochs, batch_size=batch_size)