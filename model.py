import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten, Lambda, Dropout, concatenate
from keras.optimizers import RMSprop
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

def store_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=4)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def triplet_loss(y_true, y_pred):
    alpha = 0.2

    anchor = y_pred[:,0]
    positive = y_pred[:,1]
    negative = y_pred[:,2]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=0)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=0)

    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
 
    return loss


def get_siamese_model(input_shape):
    """
        Model architecture
    """
    
    # Define the tensors for the two input images
    anchor_input = Input(input_shape, name = 'Anchor')
    pos_input = Input(input_shape, name = 'Positive')
    neg_input = Input(input_shape, name = 'Negative')
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10,10), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7,7), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,4), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,4), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='sigmoid'))

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    anchor_net = model(anchor_input)
    pos_net = model(pos_input)
    neg_net = model(neg_input)
    output = concatenate([anchor_net, pos_net, neg_net], axis=-1)

    siamese_net = Model(inputs=[anchor_input, pos_input, neg_input], outputs=output)

    # return the model
    return siamese_net

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean((1 - y_true) * square_pred + (y_true) * margin_square)

def get_siamese_model_2(input_shape):
    """
        Model architecture
    """
    
    # Define the tensors for the two input images
    left_input = Input(input_shape, name = 'LeftInput')
    right_input = Input(input_shape, name = 'RigthInput')

    # Network layers
    model = Sequential()
    model.add(Conv2D(64, (10,10), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7,7), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,4), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,4), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(euclidean_distance)
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='softmax')(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input], outputs=prediction)
    
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
            # Anchor sample
            anchor = c[i % num_samples]

            # Same-class sample
            pos = c[(i + 1) % num_samples]

            # Different-class sample
            inc = random.randrange(1, num_classes)
            icn = (num_c + inc) % num_classes
            if num_samples == n:
                neg = classes[icn][len_classes[icn] - 1 - (i % len_classes[icn])]
            else:
                neg = classes[icn][i % len_classes[icn]]
            
            pairs.append([anchor, pos, neg])

            # Labels (we don't care about the labels)
            labels.append(0)
        num_c += 1
    return np.array(pairs), np.array(labels)

def make_pairs_2(X, Y):
    classes = [X[Y==i] for i in range(num_classes)]
    len_classes = [c.shape[0] for c in classes]
    pairs = []
    labels = []

    n = min(len_classes)
    

    num_c = 0
    for c, num_samples in zip(classes, len_classes) :
        for i in range(n):
            # Same-class sample
            z1, z2 = c[i % num_samples], c[(i + 1) % num_samples]
            pairs.append([z1, z2])
            # Different-class sample
            inc = random.randrange(1, num_classes)
            icn = (num_c + inc) % num_classes
            if num_samples == n:
                z3, z4 = c[i % num_samples], classes[icn][len_classes[icn] - 1 - (i % len_classes[icn])]
            else:
                z3, z4 = c[i % num_samples], classes[icn][i % len_classes[icn]]
            pairs.append([z3, z4])
            # Labels
            labels += [0, 1]
        num_c += 1
    return np.array(pairs), np.array(labels)

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal((1 - y_true), K.cast(y_pred < 0.5, y_true.dtype)))

if __name__ == "__main__":

    print("Unpickling...")
    train = get_data("train_app1.p")
    val = get_data("val_app1.p")
    test = get_data("test_app1.p")
    print("Unpickled!")

    a = 0
    i = 0
    while a != 'q':
        trainxlist = train['X']
        trainxarray = trainxlist[i]
        trainxarray = trainxarray.reshape((512,512))
        im = Image.fromarray(trainxarray, mode='L')
        im.show()
        i += 1
        a = input()
    exit()

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
    model = get_siamese_model_2((X_train.shape[1], X_train.shape[2], 1))
    model.compile(optimizer='adam', loss=contrastive_loss, metrics=[accuracy])
    model.summary()

    
    recompute_pairs = True
    if recompute_pairs:

        print("Making pairs...")
        X_train_pairs, y_train_pairs = make_pairs_2(X_train, y_train)
        
        X_val_pairs, y_val_pairs = make_pairs_2(X_test, y_test)
        print("Pairs made")
        
        # print("Storing pairs")
        # store_object((X_train_pairs, y_train_pairs, X_val_pairs, y_val_pairs), 'pairs.p')
        # print("Pairs stored")
        
    else:
        print("Unpickling pairs...")
        X_train_pairs, y_train_pairs, X_val_pairs, y_val_pairs = get_data('pairs.p')
        print("Pairs unpickled")

    print(X_train_pairs.shape)
    print(y_train_pairs.shape)
    
    epochs = 10
    batch_size = 5
    model.fit([X_train_pairs[:,0], X_train_pairs[:,1]], y_train_pairs, epochs=epochs, batch_size=batch_size, validation_data=([X_val_pairs[:,0], X_val_pairs[:,1]], y_val_pairs), shuffle=False)
    # model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val))