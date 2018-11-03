from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm(inputs, training):
    """
    Perform batch normalization.
    """
    return tf.layers.batch_normalization(inputs=inputs, axis=3, momentum=_BATCH_NORM_DECAY,
                                         epsilon=_BATCH_NORM_EPSILON, center=True, scale=True,
                                         training=training, fused=True)


def fixed_padding(inputs, kernel_size):
    """
    Padding function.
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0, 0],
                                    [pad_beg, pad_end],
                                    [pad_beg, pad_end],
                                    [0, 0]],
                           'REFLECT')
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides):
    """
    Perform 2D convolution using fixed padding.
    """
    inputs = fixed_padding(inputs, kernel_size)

    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                            padding='VALID', use_bias=True, kernel_initializer=tf.variance_scaling_initializer(),
                            data_format='channels_last')


def get_fnet_feature_map():
    return [32, 64, 128, 256, 512]


def get_dnet_feature_map():
    return [1024]


class FeatureNet(object):
    """
    Create an image-to-feature mapping function.
    """
    def __init__(self, name, is_train):
        self.name = name
        self.is_train = is_train
        self.reuse = False

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            # Save end_points for further reuse
            end_points = {}
            outputs = inputs

            # Perform convolution + pooling
            feature_maps = get_fnet_feature_map()
            for i, n_features in enumerate(feature_maps):
                outputs = conv2d_fixed_padding(inputs=outputs, filters=n_features, kernel_size=4, strides=2)
                outputs = batch_norm(outputs, training=self.is_train)
                outputs = tf.nn.leaky_relu(outputs)
                end_points['feature_conv_%s' % i] = outputs

            # Set reuse variables to True and recover the variable list
            self.reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

            return end_points, outputs


class DecisionNet(object):
    """
    Compute decision based on input features from images A and B.
    """
    def __init__(self, name, is_train):
        self.name = name
        self.is_train = is_train
        self.reuse = False

    def __call__(self, inputs_a, inputs_b, concat=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            # Save end_points for further reuse
            end_points = {}

            # Input concatenation
            if concat:
                outputs = tf.concat([inputs_a, inputs_b], axis=3)
            else:
                outputs = inputs_a

            # Perform convolution + pooling
            feature_maps = get_dnet_feature_map()
            for i, n_features in enumerate(feature_maps):
                outputs = conv2d_fixed_padding(inputs=outputs, filters=n_features, kernel_size=4, strides=2)
                outputs = batch_norm(outputs, training=self.is_train)
                outputs = tf.nn.leaky_relu(outputs)
                end_points['decision_conv_%s' %i] = outputs

            # Add fully-connected layer
            outputs = tf.layers.conv2d(inputs=outputs, filters=1, kernel_size=outputs.get_shape().as_list()[1],
                                       padding='VALID', kernel_initializer=tf.variance_scaling_initializer(),
                                       data_format='channels_last')
            outputs = tf.squeeze(outputs, [1, 2])
            outputs = tf.nn.relu(outputs)

            # Set reuse variables to True and recover the variable list
            self.reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

            return end_points, outputs


class EarlyFusion(object):
    """
    Create an early fusion network.
    """
    def __init__(self, name, is_train):
        self.name = name
        self.is_train = is_train
        self.reuse = False

    def __call__(self, inputs_a, inputs_b):
        with tf.variable_scope(self.name, reuse=self.reuse):
            # Save end_points for further reuse
            end_points = {}

            # Input concatenation
            inputs = tf.concat([inputs_a, inputs_b], axis=3)

            # Extract features
            fnet = FeatureNet(name='feature_net', is_train=self.is_train)
            end_points_fab, outputs = fnet(inputs=inputs)
            end_points.update(end_points_fab)

            # Take a decision
            dnet = DecisionNet(name='decision_net', is_train=self.is_train)
            end_points_dab, outputs = dnet(inputs_a=outputs, inputs_b=outputs, concat=False)
            end_points.update(end_points_dab)

            self.reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

            return end_points, outputs


class LateFusion(object):
    """
    Create a late fusion network.
    """
    def __init__(self, name, is_train):
        self.name = name
        self.is_train = is_train
        self.reuse = False

    def __call__(self, inputs_a, inputs_b):
        with tf.variable_scope(self.name, reuse=self.reuse):
            # Save end_points for further reuse
            end_points = {}

            # Create feature extractor and decision networks
            fnet = FeatureNet(name='feature_net', is_train=self.is_train)
            dnet = DecisionNet(name='decision_net', is_train=self.is_train)

            end_points_a, outputs_a = fnet(inputs=inputs_a)
            end_points_b, outputs_b = fnet(inputs=inputs_b)

            # Change dict names before updating
            list_a = [key for key in end_points_a]
            for key in list_a:
                end_points_a[key + '_a'] = end_points_a.pop(key)

            list_b = [key for key in end_points_b]
            for key in list_b:
                end_points_b[key + '_b'] = end_points_b.pop(key)

            end_points.update(end_points_a)
            end_points.update(end_points_b)

            end_points_c, outputs_c = dnet(inputs_a=outputs_a, inputs_b=outputs_b, concat=True)
            end_points.update(end_points_c)

            self.reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

            return end_points, outputs_c
