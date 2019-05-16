from __future__ import absolute_import

# import the necessary packages
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .config import cifar100_config as config
from external_dir.preprocessing import ImageToArrayPreprocessor
from external_dir.preprocessing import SimplePreprocessor
from external_dir.preprocessing import MeanPreprocessor
from external_dir.io import HDF5DatasetGenerator, HDF5DatasetIterator
import json
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


def get_generator(self):
    """Loads cifar100

    # Returns
        Tuple of HDF% Generators: `(x_train, y_train), (x_test, y_test)`.

    # Raises
        ValueError: in case of invalid `label_mode`.
    """
    # load the RGB means for the training set
    means = json.loads(open(config.DATASET_MEAN).read())
    width = 32
    height = 32
    depth = 3

    # initialize the image preprocessors
    #sp = SimplePreprocessor(width, height) # Resizes image
    #mp = MeanPreprocessor(means["R"], means["G"], means["B"]) # Subtracts mean and merges in BGR order
    #iap = ImageToArrayPreprocessor() # Converts PIL to numpy

    # construct the image generator for data augmentation
    zca_batches = 50
    zca_gen = HDF5DatasetIterator(config.TRAIN_HDF5, zca_batches,
                                   nb_code_bits=config.NUM_CODE_BITS)

    zca_tr = np.vstack(next(zca_gen)[0] for _ in range(zca_batches))
    
    
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(
                samplewise_center=True,  # set each sample mean to 0
                samplewise_std_normalization=True,  # divide each input by its std
                zca_whitening=True,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.125,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.125,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False,  # randomly flip images
                fill_mode = 'nearest')

    aug.fit(zca_tr)

    # initialize the training and validation dataset generators
    '''
    trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, self.batch_size,
            aug=aug,
            preprocessors=[sp, mp, iap], nb_code_bits=config.NUM_CODE_BITS)
    valGen = HDF5DatasetGenerator(config.VAL_HDF5, self.batch_size, 
            preprocessors=[sp, mp, iap], nb_code_bits=config.NUM_CODE_BITS)
    '''


    trainGen = HDF5DatasetIterator(config.TRAIN_HDF5, self.batch_size,
                                   aug=aug,
                                   nb_code_bits=config.NUM_CODE_BITS)
    testGen = HDF5DatasetIterator(config.TEST_HDF5, self.batch_size,
                                  aug=aug,                                  
                                  nb_code_bits=config.NUM_CODE_BITS)

    info = "Data generators for hdf5 data and for augmentation"

    return (trainGen, testGen, info, (depth, height, width))
