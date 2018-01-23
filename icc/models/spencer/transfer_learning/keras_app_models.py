# -*- coding: utf-8 -*-
"""
Inspired from sources:
https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

from keras.models import Model, Sequential
from keras.layers import Conv2D, Activation, Dropout, Flatten, Dense, Input

# Local imports
import adjust_path  # Before doing any local imports
from icc.data_loader import DataLoader
from icc.contrib.preprocessing.utils import *


def build_top_VGG_model(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


class AppModel(object):

    def __init__(self, basenet: str='VGG16'):
        self._basenet = basenet


    def save_bottleneck_features(self, X, y, filename=None):

        # Build base network
        if self._basenet == 'VGG16':
            from keras.applications import VGG16
            model = VGG16(input_shape=(75,75,3), include_top=False, weights='imagenet')
        elif self._basenet == 'ResNet50':
            from keras.applications import ResNet50
            model = ResNet50(input_shape=(75,75,3), include_top=False, weights='imagenet')
        print("=> {} model loaded./n".format(self._basenet))

        # Compute bottleneck features from model.
        features = []
        for batch in batch_generator(X, batch_size=500):
            features.append(model.predict(batch))
        features = np.vstack(features)

        if filename:
            fname = filename
        else:
            fname = '../data/{}-bottleneck-features.pickle'.format(self._basenet)

        save_to_pickle([features, y], fname)
        print('=> Features saved to .. {}/n'.format(fname))
        self._path_to_bnfeats = fname


    def set_optimizer(self, opt, lr):
        if opt == 'adam':
            from keras.optimizers import Adam
            return Adam(lr=lr)
        if opt == 'nadam':
            from keras.optimizers import Nadam
            return Nadam(lr=lr)


    def run_transferlearning(self, learning_rate: float=0.0001, epochs: int=100, batch_size: int=64,
        valid_split: float=0.1, opt: str='adam', bottleneck_features_path=None, top_weights_path=None):
        """
        """
        # Load bottleneck features.
        if bottleneck_features_path:
            fname = bottleneck_features_path
        else:
            fname = '../data/{}-bottleneck-features.pickle'.format(self._basenet)
        feats, labels = load_from_pickle(fname)

        # Build a top classifier model to put on top of the convolutional VGG model.
        if self._basenet == 'VGG16':
            top_model = build_top_VGG_model(input_shape=feats.shape[1:])

        top_model.compile(optimizer=self.set_optimizer(opt, learning_rate),
         loss='binary_crossentropy', metrics=['accuracy'])
        
        top_model.fit(feats, labels, 
            validation_split=valid_split, 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=True)

        if top_weights_path:
            fname = top_weights_path
        else:
            fname = './saved_model/{}-top-model-weights.hdf5'.format(self._basenet)
        top_model.save(fname)
        print('=> Top model weights saved to .. {}\n'.format(fname))
        self._top_model_weights = fname 


    def run_finetuning(self, X, y, learning_rate: float=0.0001, epochs: int=100, batch_size: int=64, 
        valid_split: float=0.1, opt: str='adam', top_weights_path=None):
        """
        """
        # Build base network
        input_tensor = Input(shape=(75,75,3), name='image_input')

        if self._basenet == 'VGG16':
            from keras.applications import VGG16
            base_model = VGG16(input_tensor=input_tensor, include_top=False, weights='imagenet')
            print('=> VGG model loaded.\n')

            top_model = build_top_VGG_model(base_model.output_shape[1:])
            top_model.load_weights(top_weights_path)
            print('=> Top model loaded.\n')

            # Stack base and top models.
            output = top_model(base_model.output)
            self.stacked_model = Model(inputs=base_model.input, outputs=output)

            # Set first 25 layers to non-trainable
            for layer in self.stacked_model.layers[:25]:
                layer.trainable = False

            self.stacked_model.compile(optimizer=self.set_optimizer(opt, learning_rate), 
                loss='binary_crossentropy', metrics=['accuracy'])

            self.stacked_model.fit(X, y, 
                validation_split=valid_split, 
                epochs=epochs, 
                batch_size=batch_size, 
                verbose=True)

        elif self._basenet == 'ResNet50':
            # !!!!TODO: Need to FIX: apply zero padding 75x75, limit of input image is 197x197
            from keras.applications import ResNet50
            base_model = ResNet50(input_tensor=input_tensor, include_top=False, weights='imagenet')