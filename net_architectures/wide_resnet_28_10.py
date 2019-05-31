from keras.layers import Input
from keras.layers import Add
from net_architectures.sgActivation import Activation
#from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K

weight_decay = 0.0005

def initial_conv(input):
    x = Conv2D(16, (3, 3), padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=False)(input)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1,
                           epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    return x


def expand_conv(init, base, k, strides=(1, 1)):
    x = Conv2D(base * k, (3, 3), padding='same',
                      strides=strides, kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=False)(init)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1,
                           epsilon=1e-5,
                           gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = Conv2D(base * k, (3, 3), padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    skip = Conv2D(base * k, (1, 1), padding='same',
                         strides=strides,
                         kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=False)(init)

    m = Add()([x, skip])

    return m


def conv1_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1,
                           epsilon=1e-5,
                           gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Conv2D(16 * k, (3, 3), padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1,
                           epsilon=1e-5,
                           gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Conv2D(16 * k, (3, 3), padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    m = Add()([init, x])
    return m

def conv2_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1,
                           epsilon=1e-5,
                           gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Conv2D(32 * k, (3, 3), padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1,
                           epsilon=1e-5,
                           gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Conv2D(32 * k, (3, 3), padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    m = Add()([init, x])
    return m

def conv3_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1,
                           epsilon=1e-5,
                           gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Conv2D(64 * k, (3, 3), padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1,
                           epsilon=1e-5,
                           gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Conv2D(64 * k, (3, 3), padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    m = Add()([init, x])
    return m

def WideResNet(input_shape,
               nb_output_nodes,
               output_activation):
    """
    Creates a Wide Residual Network with specified parameters

    :param input: Input Keras object
    :param nb_output_nodes: Number of output nodes
    :param N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param k: Width of the network.
    :param dropout: Adds dropout if value is greater than 0.0
    :return:
    """

    # At some point turn these back into input parameters
    N=2
    k=2
    dropout=0.0
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    ip = Input(shape=input_shape)

    x = initial_conv(ip)
    x = expand_conv(x, 16, k)

    for i in range(N - 1):
        x = conv1_block(x, k, dropout)

    x = BatchNormalization(axis=channel_axis, momentum=0.1,
                           epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 32, k, strides=(2, 2))

    for i in range(N - 1):
        x = conv2_block(x, k, dropout)


    x = BatchNormalization(axis=channel_axis, momentum=0.1,
                           epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 64, k, strides=(2, 2))


    for i in range(N - 1):
        x = conv3_block(x, k, dropout)


    x = BatchNormalization(axis=channel_axis, momentum=0.1,
                           epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)

    #x = Dense(nb_output_nodes, kernel_regularizer=l2(weight_decay),
    #          activation=output_activation)(x)

    x = Dense(nb_output_nodes, kernel_regularizer=l2(weight_decay))(x)
    x = Activation(output_activation)(x)

    model = Model(ip, x)
    return model


def build_architecture(input_shape,
                       nb_output_nodes,
                       output_activation):

    model = WideResNet(input_shape=input_shape,
                       nb_output_nodes=nb_output_nodes,
                       output_activation=output_activation)

    return model
                     

