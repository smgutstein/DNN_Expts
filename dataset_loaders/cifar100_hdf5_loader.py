from __future__ import absolute_import
import os
import pickle
import sys
file_dir = os.path.dirname(os.path.realpath(__file__))
if file_dir not in sys.path:
    sys.path.append(file_dir)
file_dir2 = os.path.join(file_dir, "hdf_classes")
if file_dir2 not in sys.path:
        sys.path.append(file_dir2)

from hdf5datasetgenerator import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from os.path import expanduser
home = expanduser("~")


def load_data(self, trvate='train', label_mode='fine'):
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

    # Create HDF5 Generator
    path = os.path.join(home, self.data_path)
    fpath = os.path.join(path, trvate + '.hdf5')
    image_gen = ImageDataGenerator(**self.ImageDataGen_args)
    class_nums = pickle.load(open(os.path.join(path, 'classnums.pkl'), 'rb'))
    hdf_gen = HDF5DatasetGenerator(fpath, self.batch_size,
                                  self.nb_code_bits, image_gen)

    # Get shape of input images
    self.get_input_shape(hdf_gen.imgShape)

    # Get Encoding Dict
    self.get_encoding_dict(class_nums)
    hdf_gen.set_encoding_dict(self.encoding_dict)
    
    '''
    ############################################################
    # Test creating data_gen using info stored in hdf5 database 
    # (Sample trial got 5 epochs 42.95% val acc)
    x = hdf_gen.db["images"]
    y_classnum = hdf_gen.db["labels"]
    y_classnum = np.reshape(y_classnum, (len(y_classnum), 1))

    # Encode Class Numbers
    y_temp = y_classnum.ravel()
    Y_encoded = np.empty((len(y_temp), hdf_gen.nb_code_bits))
    for i in range(len(y_temp)):
        Y_encoded[i, :] = hdf_gen.encoding_dict[y_temp[i]]

    image_gen.fit(x)
    data_gen = image_gen.flow(x, Y_encoded, hdf_gen.batchSize)
    return data_gen, hdf_gen.batches_per_epoch
    ############################################################
    '''

    return hdf_gen.generator(), hdf_gen.batches_per_epoch



