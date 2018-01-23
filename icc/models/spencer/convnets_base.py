# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Conv2D, MaxPooling2D, Dense
from keras.layers import Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Nadam, Adam


def convnetBlue():

    model = Sequential()

    # Convolutional layer block 1
    model.add(Conv2D(filters=64, 
                     kernel_size=(3, 3), 
                     strides=(1, 1), 
                     padding="valid", 
                     input_shape=(75, 75, 3))) # change from rgb to gray scale
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Convolutional layer block 2
    model.add(Conv2D(filters=128, 
                     kernel_size=(3, 3), 
                     strides=(1, 1), 
                     padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.4))

    # Convolutional layer block 3
    model.add(Conv2D(filters=128, 
                     kernel_size=(3, 3), 
                     strides=(1, 1), 
                     padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.4))

    # Convolutional layer block 4
    model.add(Conv2D(filters=256, 
                     kernel_size=(3, 3), 
                     strides=(1, 1), 
                     padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Flatten before entering dense layers. This does not affect the batch size.
    model.add(Flatten())

    # Dense layer 1
    model.add(Dense(units=1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # Dense layer 2
    model.add(Dense(units=256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Final layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    adam_opt = Nadam(lr=0.0001, epsilon=1e-8)
    model.compile(optimizer=adam_opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def convnetWhite():

    model =Sequential()

    # Conv block 1
    model.add(Conv2D(64, kernel_size=(3,3), input_shape=(75, 75, 3)))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(64, kernel_size=(3,3)))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(64, kernel_size=(3,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Conv block 2
    model.add(Conv2D(128, kernel_size=(3,3)))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, kernel_size=(3,3)))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, kernel_size=(3,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Conv block 3
    model.add(Conv2D(128, kernel_size=(3,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    # Flatten before dense
    model.add(Flatten())

    #Dense 1
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.6))

    #Dense 2
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    # Output 
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.0001, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


def convnetRed():

    model=Sequential()
    
    # Conv block 1
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(75, 75, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu' ))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
   
    # Conv block 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    # Conv block 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    #Conv block 4
 #   model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
 #   model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    # Flatten before dense
    model.add(Flatten())

    #Dense 1
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.6))

    #Dense 2
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    # Output 
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.0001, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


def convnetGreen():

    model = Sequential()

    # Convolutional layer block 1
    model.add(Conv2D(filters=64, 
                     kernel_size=(3, 3), 
                     strides=(1, 1), 
                     padding="valid", 
                     input_shape=(75, 75, 3))) # change from rgb to gray scale
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Convolutional layer block 2
    model.add(Conv2D(filters=128, 
                     kernel_size=(3, 3), 
                     strides=(1, 1), 
                     padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.4))

    # Convolutional layer block 3
    model.add(Conv2D(filters=128, 
                     kernel_size=(3, 3), 
                     strides=(1, 1), 
                     padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.4))

    # Flatten before entering dense layers. This does not affect the batch size.
    model.add(Flatten())

    # Dense layer 1
    model.add(Dense(units=1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # Dense layer 2
    model.add(Dense(units=256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Final layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    adam_opt = Nadam(lr=0.0001, epsilon=1e-8)
    model.compile(optimizer=adam_opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model