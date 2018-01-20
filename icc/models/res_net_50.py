
import pandas as pd

from keras import layers
from keras.layers import Input
from keras.layers import Add
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import ZeroPadding2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.utils import layer_utils
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from icc.data_loader import DataLoader
from icc.models.julian.julian_base_model_keras import JBaseKerasModel

from icc.ml_stack import StackedClassifier
import os

@StackedClassifier.register
# ResNet50 residual network as used in coursera
class ResNet50 (JBaseKerasModel):

    def __init__(self, epochs = 50, batch_size = 24, weights_path = None):
        super().__init__(epochs = epochs, batch_size = batch_size, weights_path = None) #"resnet50.h5")

    def _path_to_weights(self):
        """
        Return path to weights relative to model file.
        """
        current = os.path.dirname(__file__)
        return os.path.join(current, self.weights_path)


    def _get_loss_func(self):
        return 'categorical_crossentropy'

    def _get_optimizer(self):
        return "adam" #Adam(lr = 0.001, epsilon = 1e-8)

    def _get_callbacks(self):
        callbacks = []
        filepath = 'weights.EPOCH{epoch:02d}-VAL_LOSS{val_loss:.2f}.hdf5'
        modelCheckpointCallback = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='auto', period=1)
        callbacks.append(modelCheckpointCallback)
        
        earlyStoppingCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        #callbacks.append(earlyStoppingCallback)

        return callbacks

    def get_model(self, input_shape = (75, 75, 3), classes = 2):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """
        # Define the input as a tensor with shape input_shape
        X_input = Input(input_shape)

        
        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)
        
        # Stage 1
        X = Conv2D(64, 
                   (7, 7), 
                   strides = (2, 2), 
                   name = 'conv1', 
                   kernel_initializer = glorot_uniform(seed=123))(X)
        X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = self._convolutional_block(X, 
                                f = 3, 
                                filters = [64, 64, 256], 
                                stage = 2, 
                                block='a', 
                                s = 1)
        X = self._identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = self._identity_block(X, 3, [64, 64, 256], stage=2, block='c')

        # Stage 3 
        X = self._convolutional_block(X, 
                                f = 3, 
                                filters = [128, 128, 512],
                                stage = 3,
                                block = 'a', 
                                s = 2)
        X = self._identity_block(X, 3, [128, 128, 512], stage = 3, block = 'b')
        X = self._identity_block(X, 3, [128, 128, 512], stage = 3, block = 'c')
        X = self._identity_block(X, 3, [128, 128, 512], stage = 3, block = 'd')

        # Stage 4 
        X = self._convolutional_block(X,
                               f = 3,
                               filters = [256, 256, 1024],
                               stage = 4,
                               block = 'a',
                               s = 2)
        X = self._identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'b')
        X = self._identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'c')
        X = self._identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'd')
        X = self._identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'e')
        X = self._identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'f')

        # Stage 5
        X = self._convolutional_block(X,
                               f = 3,
                               filters = [512, 512, 2048],
                               stage = 5,
                               block = 'a',
                               s = 2)
        X = self._identity_block(X, 3, [512, 512, 2048], stage = 5, block = 'b')
        X = self._identity_block(X, 3, [512, 512, 2048], stage = 5, block = 'c')

        # AVGPOOL
        X = AveragePooling2D(pool_size=(2, 2), 
                             name = 'avg_pool') (X)

        # output layer
        X = Flatten()(X)
        X = Dense(  classes, 
                    activation='softmax', 
                    name='FC' + str(classes), 
                    kernel_initializer = glorot_uniform(seed=123))(X)
        
        
        # Create model
        model = Model(  inputs = X_input, 
                        outputs = X, 
                        name='ResNet50')

        return model

    # Identity block as used in coursera
    def _identity_block(self, X, f, filters, stage, block):
        """
        Implementation of the identity block
        
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        
        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """
        
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Save the input value. You'll need this later to add back to the main path. 
        X_shortcut = X
        
        # First component of main path
        X = Conv2D(filters = F1, 
                   kernel_size = (1, 1), 
                   strides = (1,1), 
                   padding = 'valid', 
                   name = conv_name_base + '2a', 
                   kernel_initializer = glorot_uniform(seed=123))(X)
        X = BatchNormalization(axis = 3, 
                               name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)
        
        # Second component of main path
        X = Conv2D(filters = F2,
                  kernel_size = (f, f),
                  strides = (1, 1),
                  padding = 'same',
                  name = conv_name_base + '2b',
                  kernel_initializer = glorot_uniform(seed = 123)) (X)
        X = BatchNormalization(axis = 3, 
                               name = bn_name_base + '2b') (X)
        X = Activation('relu') (X)

        # Third component of main path 
        X = Conv2D(filters = F3,
                  kernel_size = (1, 1),
                  strides = (1, 1),
                  padding = 'valid',
                  name = conv_name_base + '2c',
                  kernel_initializer = glorot_uniform(seed = 123)) (X)
        X = BatchNormalization(axis = 3, 
                               name = bn_name_base + '2c') (X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation 
        X = Add()([X_shortcut, X])
        X = Activation('relu') (X)
        
        return X

    # Convolutional block as used in coursera
    def _convolutional_block(self, X, f, filters, stage, block, s = 2):
        """
        Implementation of the convolutional block
        
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        s -- Integer, specifying the stride to be used
        
        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """
        
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Save the input value
        X_shortcut = X

        # First component of main path 
        X = Conv2D(F1, 
                   (1, 1), 
                   strides = (s,s),
                   padding = 'valid',
                   name = conv_name_base + '2a', 
                   kernel_initializer = glorot_uniform(seed=123))(X)
        X = BatchNormalization(axis = 3, 
                               name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path
        X = Conv2D(filters = F2,
                  kernel_size = (f, f),
                  strides = (1, 1),
                  padding = 'same',
                  name = conv_name_base + '2b',
                  kernel_initializer = glorot_uniform(seed = 123)) (X)
        X = BatchNormalization(axis = 3, 
                               name = bn_name_base + '2b') (X)
        X = Activation('relu') (X)

        # Third component of main path
        X = Conv2D(filters = F3,
                   kernel_size = (1, 1),
                   strides = (1, 1),
                   padding = 'valid',
                   name = conv_name_base + '2c',
                   kernel_initializer = glorot_uniform(seed = 123)) (X)
        X = BatchNormalization(axis = 3, 
                               name = bn_name_base + '2c') (X)

        # Shortcut path
        X_shortcut = Conv2D(filters = F3,
                           kernel_size = (1, 1),
                           strides = (s, s),
                           padding = 'valid',
                           name = conv_name_base + '1',
                           kernel_initializer = glorot_uniform(seed = 123)) (X_shortcut)
        X_shortcut = BatchNormalization(axis = 3, 
                                        name = bn_name_base + '1') (X_shortcut)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = Add() ([X_shortcut, X])
        X = Activation('relu') (X)
        
        return X