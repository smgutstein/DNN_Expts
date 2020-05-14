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
import  net_architectures.wide_residual_network_base as wrn


def build_architecture(input_shape,
                       nb_output_nodes,
                       net_params):
    """Instantiates the wide resnet architecture."""
    # Default Values
    N = 4
    k = 10
    dropout = 0.00

    # User Values
    if 'n' in net_params:
        N = net_params['n']

    if 'k' in net_params:
        k = net_params['k']
    
    if 'dropout' in net_params:
        dropout = net_params['dropout']
    

    model = wrn.create_wide_residual_network(input_shape,
                                             nb_classes=nb_output_nodes,
                                             N=N, k=k,
                                             dropout=dropout)



    return model
