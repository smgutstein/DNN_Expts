from __future__ import absolute_import

# import the necessary packages
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .config import tiny_image_net_config as config
from external_dir.preprocessing import ImageToArrayPreprocessor
from external_dir.preprocessing import SimplePreprocessor
from external_dir.preprocessing import MeanPreprocessor
from external_dir.io import HDF5DatasetGenerator, HDF5DatasetIterator
import json
from keras.preprocessing.image import ImageDataGenerator


def get_generator(self):
    """Loads tiny imagenet

    # Returns
        Tuple of HDF% Generators: `(x_train, y_train), (x_test, y_test)`.

    # Raises
        ValueError: in case of invalid `label_mode`.
    """
    # load the RGB means for the training set
    means = json.loads(open(config.DATASET_MEAN).read())
    width = 64
    height = 64
    depth = 3

    # initialize the image preprocessors
    sp = SimplePreprocessor(width, height)
    mp = MeanPreprocessor(means["R"], means["G"], means["B"])
    iap = ImageToArrayPreprocessor()

    # construct the image generator for data augmentation
    aug = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1, horizontal_flip=True,
                             fill_mode="nearest")

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
            preprocessors=[sp, mp, iap], nb_code_bits=config.NUM_CODE_BITS)
    valGen = HDF5DatasetIterator(config.VAL_HDF5, self.batch_size,
            preprocessors=[sp, mp, iap], nb_code_bits=config.NUM_CODE_BITS)

    info = "Data generators for hdf5 data and for augmentation"

    return (trainGen, valGen, info, (depth, height, width))
