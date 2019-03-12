# -*- coding: utf-8 -*-
"""Googlenet model for Keras.


"""
from __future__ import print_function
from __future__ import absolute_import
import keras
import warnings

# import the necessary packages
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras import backend as K

class MiniGoogLeNet:
	@staticmethod
	def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):
		# define a CONV => BN => RELU pattern
		x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Activation("relu")(x)

		# return the block
		return x

	@staticmethod
	def inception_module(x, numK1x1, numK3x3, chanDim):
		# define two CONV modules, then concatenate across the
		# channel dimension
		conv_1x1 = MiniGoogLeNet.conv_module(x, numK1x1, 1, 1,
			(1, 1), chanDim)
		conv_3x3 = MiniGoogLeNet.conv_module(x, numK3x3, 3, 3,
			(1, 1), chanDim)
		x = concatenate([conv_1x1, conv_3x3], axis=chanDim)

		# return the block
		return x

	@staticmethod
	def downsample_module(x, K, chanDim):
		# define the CONV module and POOL, then concatenate
		# across the channel dimensions
		conv_3x3 = MiniGoogLeNet.conv_module(x, K, 3, 3, (2, 2),
			chanDim, padding="valid")
		pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
		x = concatenate([conv_3x3, pool], axis=chanDim)

		# return the block
		return x

	@staticmethod
	def build(input_shape, nb_output_nodes, output_activation):

		# set channel dimension based on K.image_data_format
		if K.image_data_format() == "channels_first":
		    chanDim = 1
                else:
		    chanDim = -1

                    
		# define the model input and first CONV module
		inputs = Input(shape=input_shape)
		x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1),
			chanDim)

		# two Inception modules followed by a downsample module
		x = MiniGoogLeNet.inception_module(x, 32, 32, chanDim)
		x = MiniGoogLeNet.inception_module(x, 32, 48, chanDim)
		x = MiniGoogLeNet.downsample_module(x, 80, chanDim)

		# four Inception modules followed by a downsample module
		x = MiniGoogLeNet.inception_module(x, 112, 48, chanDim)
		x = MiniGoogLeNet.inception_module(x, 96, 64, chanDim)
		x = MiniGoogLeNet.inception_module(x, 80, 80, chanDim)
		x = MiniGoogLeNet.inception_module(x, 48, 96, chanDim)
		x = MiniGoogLeNet.downsample_module(x, 96, chanDim)

		# two Inception modules followed by global POOL and dropout
		x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
		x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
		x = AveragePooling2D((7, 7))(x)
		x = Dropout(0.5)(x)

		# softmax classifier
		x = Flatten()(x)
		x = Dense(nb_output_nodes)(x)
		x = Activation(output_activation)(x)

		# create the model
		model = Model(inputs, x, name="googlenet")

		# return the constructed network architecture
		return model


def build_architecture(input_shape,
                       nb_output_nodes,
                       output_activation):

    model = MiniGoogLeNet.build(input_shape, nb_output_nodes, output_activation)
    return model
