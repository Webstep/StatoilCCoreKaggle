# -*- coding: utf-8 -*-

"""AlexNet model.

Related papers:
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class AlexNetBase(object):
    """Actual implementation of AlexNet model. """

    def inference(self, minibatch_X, n_classes: int=2):
        """Infer scores from AlexNet model.

        Args:
            X: Tensor. Images with shape 75x75
            num_classes: int.  Classifying is_iceberg, default value is 2.

        Returns:
            logits: Tensor. Shape (batch_size, num_classes)
        """
        with tf.name_scope('LAYER_1'):
            convol1 = tf.layers.conv2d(
                    inputs = minibatch_X,
                    filters = 96,
                    kernel_size = [9, 9],
                    strides = (2, 2),
                    padding = "VALID",
                    activation = tf.nn.relu,
                    use_bias = True,
                    kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                    bias_initializer = tf.zeros_initializer(),
                    name = 'convol_relu_1'
                    )

            local_response_norm_1 = tf.nn.local_response_normalization(
                    convol1,
                    alpha = 2e-05,
                    beta = 0.75,
                    depth_radius = 2,
                    bias = 1.0,
                    name = 'LRNorm1'
                    )

            max_pool1 = tf.nn.max_pool(
                    local_response_norm_1,
                    ksize = [1, 3, 3, 1],
                    strides = [1, 2, 2, 1],
                    padding = 'VALID',
                    name = 'max_pool1'
                    )

        with tf.name_scope('LAYER_2'):
            convol2 = tf.layers.conv2d(
                    inputs = max_pool1,
                    filters = 256,
                    kernel_size = [5, 5],
                    strides = (1, 1),
                    padding = "SAME",
                    activation = tf.nn.relu,
                    use_bias = True,
                    kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                    bias_initializer = tf.zeros_initializer(),
                    name = 'convol_relu_2'
                    )

            local_response_norm_2 = tf.nn.local_response_normalization(
                    convol2,
                    alpha = 2e-05,
                    beta = 0.75,
                    depth_radius = 2,
                    bias = 1.0,
                    name = 'LRNorm2'
                    )

            max_pool2 = tf.nn.max_pool(
                    local_response_norm_2,
                    ksize = [1, 3, 3, 1],
                    strides = [1, 2, 2, 1],
                    padding = "VALID",
                    name = 'max_pool2'
                    )

        with tf.name_scope('LAYER_3'):
            convol3 = tf.layers.conv2d(
                    inputs = max_pool2,
                    filters = 384,
                    kernel_size = [3, 3],
                    strides = (1, 1),
                    padding = "SAME",
                    activation = tf.nn.relu,
                    use_bias = True,
                    kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                    bias_initializer = tf.zeros_initializer(),
                    name = 'convol_relu_3'
                    )

        with tf.name_scope('LAYER_4'):
            convol4 = tf.layers.conv2d(
                    inputs = convol3,
                    filters = 384,
                    kernel_size = [3, 3],
                    strides = (1, 1),
                    padding = "SAME",
                    activation = tf.nn.relu,
                    use_bias = True,
                    kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                    bias_initializer = tf.zeros_initializer(),
                    name = 'convol_relu_4'
                    )

        with tf.name_scope('LAYER_5'):
            convol5 = tf.layers.conv2d(
                    inputs = convol4,
                    filters = 256,
                    kernel_size = [3, 3],
                    strides = (1, 1),
                    padding = "SAME",
                    activation = tf.nn.relu,
                    use_bias = True,
                    kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                    bias_initializer = tf.zeros_initializer(),
                    name = 'convol_relu_5'
                    )

            max_pool5 = tf.nn.max_pool(
                    convol5,
                    ksize = [1, 3, 3, 1],
                    strides = [1, 2, 2, 1],
                    padding = "VALID",
                    name = 'max_pool5',
                    )

        def init_weights_and_biases(num_layers,layers_shape,
                    init_w_op = tf.truncated_normal_initializer(stddev = 0.1),
                    init_b_op = tf.zeros_initializer()):
            """Helper function to create weights and biases."""
            param = {}
            
            for i in range(num_layers):
                param['W%s' % i] = tf.get_variable('W%s' % i, layers_shape[i][0], initializer = init_w_op)
                param['b%s' % i] = tf.get_variable('b%s' % i, layers_shape[i][1], initializer = init_b_op)
            return param

        with tf.name_scope('fully_connected_network'):
            # We flatten the outputs after the max_pool5 operation from a 3-d matrix into 1-d because inputs 
            # into FC require a 1-d vector.
            input_shape = 3*3*256
            flat_inputs = tf.reshape(max_pool5, [-1, input_shape], name = 'flat_maxpool5_outputs')

            # Define the matrix shapes for the weights and biases of each layer. Each tuple represents a layer
            # where the first component stores the weight shape and the second component stores the bias shape.
            # For example: [(W0.shape, B0.shape), (W1.shape, B1.shape), .. ]
            layers_shape = [
                            ([input_shape, 4096], [4096]), # layer 1
                            ([4096, 4096], [4096]), # layer 2
                            ([4096, n_classes], [n_classes]) # layer 3
                            ] 

            param = init_weights_and_biases(3, layers_shape)

            with tf.name_scope('FC_layer_1'):
                with tf.name_scope('relu_activations'):
                    fc6 = tf.nn.relu(tf.matmul(flat_inputs, param['W0']) + param['b0'])

            with tf.name_scope('FC_layer_2'):
                with tf.name_scope('relu_activations'):
                    fc7 = tf.nn.relu(tf.matmul(fc6, param['W1']) + param['b1'])

            with tf.name_scope('FC_logits'):
                logits = tf.nn.xw_plus_b(fc7, param['W2'], param['b2'], name = 'logits')

        return logits