# -*- coding: utf-8 -*-
"""Core Keras layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import copy
import types as python_types
import warnings

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import InputSpec
from keras.engine.base_layer import Layer
from keras.utils.generic_utils import func_dump
from keras.utils.generic_utils import func_load
from keras.utils.generic_utils import deserialize_keras_object
from keras.utils.generic_utils import has_arg
from keras.legacy import interfaces

class Activation(Layer):
    """Applies an activation function to an output.

    # Arguments
        activation: name of activation function to use
            (see: [activations](../activations.md)),
            or alternatively, a Theano or TensorFlow operation.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    """

    def __init__(self, activation, **kwargs):
        super(Activation, self).__init__(**kwargs)
        self.supports_masking = True
        self.activation = activations.get(activation)
        self.name = self.name.replace('activation',activation)

    def call(self, inputs):
        return self.activation(inputs)

    def get_config(self):
        config = {'activation': activations.serialize(self.activation)}
        base_config = super(Activation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
