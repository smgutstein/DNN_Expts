"""VGG16 model for Keras.

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input #Activation, Flatten, Input
from net_architectures.sgActivation import Activation
from keras.layers import Conv2D, Convolution2D, MaxPooling2D, BatchNormalization


def build_architecture(input_shape,
                       nb_output_nodes,
                       output_activation,
                       **kwargs):
    """Instantiates the VGG16 architecture."""

    model = Sequential()
    inputs = Input(shape=input_shape)


    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same',
                     name='block1_conv1', input_shape = input_shape))
    model.add(Activation('relu', name='block1_relu1'))
    model.add(Conv2D(64, (3, 3), padding='same', name='block1_conv2'))
    model.add(Activation('relu', name='block1_relu2'))             
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same',
                     name='block2_conv1', input_shape = input_shape))
    model.add(Activation('relu', name='block2_relu1'))   
    model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv2'))
    model.add(Activation('relu', name='block2_relu2'))              
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same',
                     name='block3_conv1', input_shape = input_shape))
    model.add(Activation('relu', name='block3_relu1'))             
    model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv2'))
    model.add(Activation('relu', name='block3_relu2'))             
    model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv3'))
    model.add(Activation('relu', name='block3_relu3'))             
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same',
                     name='block4_conv1', input_shape = input_shape))
    model.add(Activation('relu', name='block4_relu1'))             
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv2'))
    model.add(Activation('relu', name='block4_relu2'))             
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv3'))
    model.add(Activation('relu', name='block4_relu3'))             
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), padding='same',
                     name='block5_conv1', input_shape = input_shape))
    model.add(Activation('relu', name='block5_relu1'))              
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv2'))
    model.add(Activation('relu', name='block5_relu2'))              
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv3'))
    model.add(Activation('relu', name='block5_relu3'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(512, name='fc1'))
    model.add(Activation('relu', name='fc1_relu1'))
    model.add(Dropout(0.5, name='fc1_dropout'))
    model.add(Dense(512, name='fc2'))
    model.add(Activation('relu', name='fc2_relu2'))
    model.add(Dropout(0.5, name='fc2_dropout'))
    model.add(Dense(nb_output_nodes, name="fc_prediction"))
    model.add(Activation(output_activation, name="fc_prediction_out"))

    return model
