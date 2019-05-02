from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten #Activation, Flatten
from net_architectures.sgActivation import Activation
from keras.layers import Conv2D, Convolution2D, MaxPooling2D

def build_architecture(input_shape,
                       nb_output_nodes,
                       output_activation):

    # Note: input_shape = (channels, rows, cols)
    
    model = Sequential()

    model.add(Convolution2D(32, (3, 3), padding='same',
              input_shape=input_shape))

    model.add(Activation('relu', name='relu_1'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu',name='relu_2') )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu', name='relu_3'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu', name='relu_4'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu', name='relu_5'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_output_nodes))
    model.add(Activation(output_activation, name=output_activation+'_out'))

    return model
