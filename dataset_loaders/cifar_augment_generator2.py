from __future__ import absolute_import
import os
import sys
file_dir = os.path.dirname(os.path.realpath(__file__))
if file_dir not in sys.path:
    sys.path.append(file_dir)
from cifar import load_batch
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import get_file
from keras import backend as K
import numpy as np
import types


def load_data(label_mode='fine'):
    """Loads CIFAR100 dataset.

    # Arguments
        label_mode: one of "fine", "coarse".

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    # Raises
        ValueError: in case of invalid `label_mode`.
    """
    if label_mode not in ['fine', 'coarse']:
        raise ValueError('`label_mode` must be one of `"fine"`, `"coarse"`.')

    dirname = 'cifar-100-python'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True)

    fpath = os.path.join(path, 'train')
    x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(path, 'test')
    x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # Rescale raw data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.
    x_test /= 255.

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def set_encoding_dict(self, encoding_dict):
    self.encoding_dict = encoding_dict


def get_generator(self):

    (x_train, y_train), (x_test, y_test) = load_data()    
    
    # this will do preprocessing and realtime data augmentation
    train_gen = ImageDataGenerator(
                    featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=True,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=True,  # divide each input by its std
                    zca_whitening=True,  # apply ZCA whitening
                    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.125,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.125,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=True,  # randomly flip images
                    vertical_flip=False)  # randomly flip images

    train_gen.fit(x_train)
    train_gen = train_gen.flow(x_train, y_train, batch_size=self.batch_size)
    train_gen.numImages = y_train.shape[0]
    train_gen.class_nums = sorted(list(set(np.unique(y_train))))
    train_gen.batches_per_epoch = train_gen.numImages //self.batch_size
    train_gen.set_encoding_dict = types.MethodType(set_encoding_dict,
                                                  train_gen,
                                                  ImageDataGenerator)

    # create generator for test data - like generator for train data, only without image augmentation
    val_gen = ImageDataGenerator(
                    featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=True,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=True,  # divide each input by its std
                    zca_whitening=True,  # apply ZCA whitening
                    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.0,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=False,  # randomly flip images
                    vertical_flip=False)  # randomly flip images
    info = "Data generator to augment data"

    # Get rows, cols and channels. Assume smallest dim, other than 0th
    # is channel dim
    _, temp1, temp2, temp3 = x_train.shape
    if min(temp1, temp2, temp3) == temp3:
        # Data channels last
        _, height, width, depth = x_train.shape
    elif min(temp1, temp2, temp3) == temp1:
        # Data channels first
        _, depth, height, width = x_train.shape


    val_gen.fit(x_test)
    val_gen = val_gen.flow(x_test, y_test, batch_size=self.batch_size)
    val_gen.numImages = y_test.shape[0]
    val_gen.class_nums = sorted(list(set(np.unique(y_test))))
    val_gen.batches_per_epoch = val_gen.numImages // self.batch_size
    val_gen.set_encoding_dict = types.MethodType(set_encoding_dict,
                                                val_gen,
                                                ImageDataGenerator)


    return (train_gen, val_gen, info, (depth, height, width))

