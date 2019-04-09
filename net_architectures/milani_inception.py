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
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras import backend as K


class Inception_v3:

    # set channel dimension based on K.image_data_format
    if K.image_data_format() == "channels_first":
        channel_axis = 1
    else:
        channel_axis = -1

    print("Image Format =",K.image_data_format())
    
    @staticmethod
    def conv2D_bn(x, filters, kX, kY, strides=(1, 1), padding="same"):

        # define a CONV => BN => RELU pattern
        x = Conv2D(filters, (kX, kY), strides=strides, padding=padding)(x)
        x = BatchNormalization(axis=Inception_v3.channel_axis)(x)
        x = Activation("relu")(x)

        # return the block
        return x

    @staticmethod
    def inception1(x, a_data, b_data, c_data,
                   pool_data, name):

            conv2D_bn = Inception_v3.conv2D_bn
            a_filters, a_x, a_y = a_data[0]
            branch_a = conv2D_bn(x, a_filters, a_x, a_y)
            b0_filters, b0_x, b0_y = b_data[0]
            b1_filters, b1_x, b1_y = b_data[1]
            branch_b = conv2D_bn(x, b0_filters, b0_x, b0_y)
            branch_b = conv2D_bn(branch_b, b1_filters, b1_x, b1_y)

            c0_filters, c0_x, c0_y = c_data[0]
            c1_filters, c1_x, c1_y = c_data[1]
            c2_filters, c2_x, c2_y = c_data[2]
            branch_c = conv2D_bn(x, c0_filters, c0_x, c0_y)
            branch_c = conv2D_bn(branch_c, c1_filters, c1_x, c1_y)
            branch_c = conv2D_bn(branch_c, c2_filters, c2_x, c2_y)

            pool_kernel, pool_strides, pool_padding = pool_data[0]
            pool_filters, pool_x, pool_y = pool_data[1]
            branch_pool = AveragePooling2D(pool_kernel,
                                           strides=pool_strides,
                                           padding=pool_padding)(x)
            branch_pool = conv2D_bn(branch_pool, pool_filters, pool_x, pool_y)

            x = concatenate([branch_a, branch_b,
                             branch_c, branch_pool],
                             axis=Inception_v3.channel_axis,
                             name=name)
            return x


    @staticmethod
    def inception2 (x, a_data, b_data,
                    c_data, pool_data,
                    name):

            conv2D_bn = Inception_v3.conv2D_bn
            a_filters, a_x, a_y = a_data[0]
            branch_a = conv2D_bn(x, a_filters, a_x, a_y)

            b0_filters, b0_x, b0_y = b_data[0]
            b1_filters, b1_x, b1_y = b_data[1]
            b2_filters, b2_x, b2_y = b_data[2]
            branch_b = conv2D_bn(x, b0_filters, b0_x, b0_y)
            branch_b = conv2D_bn(branch_b, b1_filters, b1_x, b1_y)
            branch_b = conv2D_bn(branch_b, b2_filters, b2_x, b2_y)

            c0_filters, c0_x, c0_y = c_data[0]
            c1_filters, c1_x, c1_y = c_data[1]
            c2_filters, c2_x, c2_y = c_data[2]
            c3_filters, c3_x, c3_y = c_data[3]
            c4_filters, c4_x, c4_y = c_data[4]
            branch_c = conv2D_bn(x, c0_filters, c0_x, c0_y)
            branch_c = conv2D_bn(branch_c, c1_filters, c1_x, c1_y)
            branch_c = conv2D_bn(branch_c, c2_filters, c2_x, c2_y)
            branch_c = conv2D_bn(branch_c, c3_filters, c3_x, c3_y)
            branch_c = conv2D_bn(branch_c, c4_filters, c4_x, c4_y)

            pool_kernel, pool_strides, pool_padding = pool_data[0]
            pool_filters, pool_x, pool_y = pool_data[1]
            branch_pool = AveragePooling2D(pool_kernel,
                                           strides=pool_strides,
                                           padding=pool_padding)(x)
            branch_pool = conv2D_bn(branch_pool, pool_filters, pool_x, pool_y)

            x = concatenate([branch_a, branch_b,
                             branch_c, branch_pool],
                             axis=Inception_v3.channel_axis,
                             name=name)
            return x

    @staticmethod
    def inception3 (x, a_data, b_data,
                    c_data, pool_data,
                    name):
            conv2D_bn = Inception_v3.conv2D_bn

            a_filters, a_x, a_y = a_data[0]
            branch_a = conv2D_bn(x, a_filters, a_x, a_y)

            b0_filters, b0_x, b0_y = b_data[0]
            b1_filters, b1_x, b1_y = b_data[1]
            b2_filters, b2_x, b2_y = b_data[2]
            branch_b = conv2D_bn(x, b0_filters, b0_x, b0_y)
            branch_b1 = conv2D_bn(branch_b, b1_filters, b1_x, b1_y)
            branch_b2 = conv2D_bn(branch_b, b2_filters, b2_x, b2_y)
            branch_b = concatenate([branch_b1, branch_b2],
                                   axis=Inception_v3.channel_axis,
                                   name=name+'_a')

            c0_filters, c0_x, c0_y = c_data[0]
            c1_filters, c1_x, c1_y = c_data[1]
            c2_filters, c2_x, c2_y = c_data[2]
            c3_filters, c3_x, c3_y = c_data[3]
            branch_c = conv2D_bn(x, c0_filters, c0_x, c0_y)
            branch_c = conv2D_bn(branch_c, c1_filters, c1_x, c1_y)
            branch_c1 = conv2D_bn(branch_c, c2_filters, c2_x, c2_y)
            branch_c2 = conv2D_bn(branch_c, c3_filters, c3_x, c3_y)
            branch_c = concatenate([branch_c1, branch_c2],
                                   axis=Inception_v3.channel_axis,
                                   name=name+'_b')


            pool_kernel, pool_strides, pool_padding = pool_data[0]
            pool_filters, pool_x, pool_y = pool_data[1]
            branch_pool = AveragePooling2D(pool_kernel,
                                           strides=pool_strides,
                                           padding=pool_padding)(x)
            branch_pool = conv2D_bn(branch_pool, pool_filters, pool_x, pool_y)

            x = concatenate([branch_a, branch_b, branch_c, branch_pool],
                             axis=Inception_v3.channel_axis, name=name)
            return x



    @staticmethod
    def downsample1(x, a_data, b_data, pool_data, name):
            conv2D_bn = Inception_v3.conv2D_bn

            a0_filters, a0_x, a0_y = a_data[0]
            a1_conv_data, a1_strides, a1_padding = a_data[1]
            a1_filters, a1_x, a1_y = a1_conv_data
            branch_a = conv2D_bn(x, a0_filters, a0_x, a0_y)
            branch_a = conv2D_bn(branch_a, a1_filters, a1_x, a1_y,
                                 strides=a1_strides,
                                 padding=a1_padding)

            b0_filters, b0_x, b0_y = b_data[0]
            b1_filters, b1_x, b1_y = b_data[1]
            b2_conv_data, b2_strides, b2_padding = b_data[2]
            b2_filters, b2_x, b2_y = b2_conv_data
            branch_b = conv2D_bn(x, b0_filters, b0_x, b0_y)
            branch_b = conv2D_bn(branch_b, b1_filters, b1_x, b1_y)
            branch_b = conv2D_bn(branch_b, b2_filters, b2_x, b2_y,
                                 strides=b2_strides,
                                 padding=b2_padding)

            pool_kernel, pool_strides, pool_padding = pool_data
            branch_pool = MaxPooling2D(pool_kernel,
                                       strides=pool_strides,
                                       padding=pool_padding)(x)

            x = concatenate([branch_a, branch_b,
                             branch_pool],
                             axis=Inception_v3.channel_axis,
                             name=name)
            return x



    @staticmethod
    def downsample2(x, a_data, b_data, pool_data, name):
            conv2D_bn = Inception_v3.conv2D_bn

            a0_filters, a0_x, a0_y = a_data[0]
            a1_filters, a1_x, a1_y, a1_strides, a1_padding = a_data[1]
            branch_a = conv2D_bn(x, a0_filters, a0_x, a0_y)
            branch_a = conv2D_bn(branch_a, a1_filters, a1_x, a1_y,
                                 strides=a1_strides,
                                 padding=a1_padding)

            b0_filters, b0_x, b0_y = b_data[0]
            b1_filters, b1_x, b1_y = b_data[1]
            b2_filters, b2_x, b2_y, b2_strides, b2_padding = b_data[2]
            branch_b = conv2D_bn(x, b0_filters, b0_x, b0_y)
            branch_b = conv2D_bn(branch_b, b1_filters, b1_x, b1_y)
            branch_b = conv2D_bn(branch_b, b2_filters, b2_x, b2_y,
                                 strides=b2_strides,
                                 padding=b2_padding)

            pool_kernel, pool_strides = pool_data[0]
            branch_pool = MaxPooling2D(pool_kernel,
                                       strides=pool_strides)(x)

            x = concatenate([branch_a, branch_b,
                             branch_pool],
                             axis=Inception_v3.channel_axis,
                             name=name)
            return x

        
    @staticmethod
    def build(input_shape, nb_output_nodes, output_activation):

                    
        # define the model input and first CONV module
        inputs = Input(shape=input_shape)

        x = Inception_v3.conv2D_bn(inputs, 32, 3, 3,
                                strides=(2, 2), padding='valid')
        x = Inception_v3.conv2D_bn(x, 32, 3, 3, padding='valid')
        x = Inception_v3.conv2D_bn(x, 64, 3, 3, padding='valid')

        # x = Inception_v3.inception1(x,[[64,1,1]],
        #                            [[48,1,1], [64,3,3]],
        #                            [[64,1,1], [96,3,3], [96,3,3]],
        #                            [[(3,3),(1,1),'same'], [32,1,1]],
        #                            'mixed0')

        #x = Inception_v3.inception1(x,[[64,1,1]],
        #                            [[48,1,1], [64,5,5]],
        #                            [[64,1,1], [96,3,3], [96,3,3]],
        #                            [[(3,3),(1,1),'same'], [64,1,1]],
        #                            'mixed1')

        # x = Inception_v3.downsample1(x,
        #                             [[192,1,1], [[384,3,3], (2,2), 'same']],
        #                             [[64,1,1], [96,3,3], [[96,3,3], (2,2), 'same']],
        #                              [(3,3),(2,2), 'same'],
        #                             'mixed3')

        # x = Inception_v3.inception2(x,
        #                            [[192,1,1]],
        #                            [[128,1,1], [128,7,1], [128,1,7],
        #                             [128,7,1], [192,1,7]],
        #                            [[(3,3),(1,1),'same'], [192,1,1]],
        #                            'mixed4')

        #x = Inception_v3.inception2(x,[[192,1,1]],
        #                            [[160,1,1], [160,1,7], [192,7,1]],
        #                            [[160,1,1], [160,7,1], [160,1,7],
        #                            [160,7,1], [192,1,7]],
        #                            [[(3,3),(1,1),'same'], [192,1,1]],
        #                            'mixed5')

        #x = Inception_v3.inception2(x,[[192,1,1]],
        #                            [[160,1,1], [160,1,7], [192,7,1]],
        #                            [[160,1,1], [160,7,1], [160,1,7],
        #                             [160,7,1], [192,1,7]],
        #                            [[(3,3),(1,1),'same'], [192,1,1]],
        #                            'mixed6')


        x = Inception_v3.inception1(x, [[36, 1, 1]],
                                    [[27, 1, 1], [36, 3, 3]],
                                    [[36, 1, 1], [54, 3, 3], [54, 3, 3]],
                                    [[(3, 3), (1, 1), 'same'], [18, 1, 1]],
                                    'mixed0')

        x = Inception_v3.inception1(x, [[36, 1, 1]],
                                [[27, 1, 1], [36, 3, 3]],
                                [[36, 1, 1], [54, 3, 3], [54, 3, 3]],
                                [[(3, 3), (1, 1), 'same'], [18, 1, 1]],
                                'mixed1')


        x = Inception_v3.downsample1(x,
                                     [[54,1,1], [[108,3,3], (2,2), 'same']],
                                     [[24,1,1], [24,3,3], [[36,3,3], (2,2), 'same']],
                                      [(3,3),(2,2), 'same'],
                                     'mixed3')


        x = Inception_v3.inception2(x,
                                    [[72,1,1]],
                                    [[48,1,1], [48,1,7], [72,7,1]],
                                    [[48,1,1], [48,7,1], [48,1,7],
                                     [48,7,1], [72,1,7]],
                                    [[(3,3),(1,1),'same'], [72,1,1]],
                                    'mixed4')

        x = Inception_v3.inception2(x,
                                    [[72,1,1]],
                                    [[48,1,1], [48,1,7], [72,7,1]],
                                    [[48,1,1], [48,7,1], [48,1,7],
                                     [48,7,1], [72,1,7]],
                                    [[(3,3),(1,1),'same'], [72,1,1]],
                                    'mixed5')

        x = Inception_v3.inception2(x,
                                    [[72,1,1]],
                                    [[48,1,1], [48,1,7], [72,7,1]],
                                    [[48,1,1], [48,7,1], [48,1,7],
                                     [48,7,1], [72,1,7]],
                                    [[(3,3),(1,1),'same'], [72,1,1]],
                                    'mixed6')



        x = Inception_v3.downsample1(x,
                                     [[124,1,1], [[186,3,3], (2,2), 'same']],
                                     [[68, 1, 1], [68, 3, 3], [[102, 3, 3], (2, 2), 'same']],
                                     [(3,3),(2,2), 'same'],
                                     'mixed8')

        x = Inception_v3.inception3(x,
                                    [[216,1,1]],
                                    [[216,1,1], [216,1,3], [216,3,1]],
                                    [[252,1,1], [216,3,3], [216,1,3], [216,3,1]],
                                    [[(3,3),(1,1),'same'], [108,1,1]],
                                    'mixed10')

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(nb_output_nodes, activation=output_activation, name='predictions')(x)

        # create the model
        model = Model(inputs, x, name="Inception_v3ish")

        # return the constructed network architecture
        return model

                

def build_architecture(input_shape,
                       nb_output_nodes,
                       output_activation):

    model = Inception_v3.build(input_shape, nb_output_nodes, output_activation)
    return model

