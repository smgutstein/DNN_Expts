from __future__ import absolute_import
import os
import sys
file_dir = os.path.dirname(os.path.realpath(__file__))
if file_dir not in sys.path:
    sys.path.append(file_dir)
from cifar import load_batch
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from os.path import expanduser
home = expanduser("~")


def load_data(self, label_mode='fine'):
    """Loads src tasks for CIFAR100 living_vs_notliving datasets.

    # Arguments
        label_mode: one of "fine", "coarse".

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    # Raises
        ValueError: in case of invalid `label_mode`.
    """
    if label_mode not in ['fine', 'coarse']:
        raise ValueError('`label_mode` must be one of `"fine"`, `"coarse"`.')

    path = os.path.join(home,'.keras/datasets/', 'cifar-100-python',
                        'cifar100_living_notliving', 'src_tasks') 

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

    # Get sorted list of class numbers (np.unique returns sorted list)
    class_nums = sorted(list(np.unique(y_train)))

    # Verbose way of changing this method to class member
    self.X_train = x_train
    self.y_train_classnum = y_train
    self.X_test = x_test
    self.y_test_classnum = y_test
    self.class_nums = class_nums

    # Note: All below involves getting/making global properties to be
    #       used with data. Current implementation assumes all data can
    #       be read in at once. After creating hdf5 generators, will
    #       see if following code can be moved to other modules, so
    #       that "load_data" will *ONLY* load data.
    # Adding encoding
    self.make_encoding_dict(**self.joint_dict)
    self.encodings = np.asarray([self.encoding_dict[x]
                                 for x in sorted(self.encoding_dict)])
    nb_outputs = self.encodings.shape[1]

    y_temp = self.y_train_classnum.ravel()
    self.Y_train_encoded = np.empty((len(y_temp), nb_outputs))
    for i in range(len(y_temp)):
        self.Y_train_encoded[i, :] = self.encoding_dict[y_temp[i]]

    y_temp = self.y_test_classnum.ravel()
    self.Y_test_encoded = np.empty((len(y_temp), nb_outputs))
    for i in range(len(y_temp)):
        self.Y_test_encoded[i, :] = self.encoding_dict[y_temp[i]]

    # Set batches (i.e. steps) per epoch
    self.train_batches_per_epoch = self.Y_train_encoded.shape[0] // self.batch_size + 1
    self.test_batches_per_epoch = self.Y_test_encoded.shape[0] // self.batch_size + 1

    # Get rows, cols and channels. Assume smallest dim, other than 0th
    # is channel dim
    _, temp1, temp2, temp3 = self.X_train.shape
    if min(temp1, temp2, temp3) == temp3:
        # Data channels last
        _, self.img_rows, self.img_cols, self.img_channels = self.X_train.shape
    elif min(temp1, temp2, temp3) == temp1:
        # Data channels first
        _, self.img_channels, self.img_rows, self.img_cols = self.X_train.shape

    # Overwrite default args for ImageDataGenerator with those from cfg files
    # (This might just be a little silly - could just combine dicts created from
    # cfg files and not explicitly state default values, but for now helps me keep
    # track of what default values are)
    for x in self.preprocess_param_dict:
        self.ImageDataGen_args[x] = self.preprocess_param_dict[x]
    for x in self.augment_param_dict:
        self.ImageDataGen_args[x] = self.augment_param_dict[x]

    print("Preprocessing Train Images")
    self.train_image_gen = ImageDataGenerator(**self.ImageDataGen_args)
    self.train_image_gen.fit(self.X_train)
    self.train_data_gen = self.train_image_gen.flow(self.X_train,
                                                    self.Y_train_encoded,
                                                    self.batch_size)

    print("Preprocessing Test Images")
    self.test_image_gen = ImageDataGenerator(**self.ImageDataGen_args)
    self.test_image_gen.fit(self.X_test)
    self.test_data_gen = self.test_image_gen.flow(self.X_test,
                                                  self.Y_test_encoded,
                                                  self.batch_size)

    return None
