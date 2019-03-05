from __future__ import absolute_import

from config import tiny_imagenet_config as config
from preprocessing import ImageToArrayPreprocessor
from preprocessing import SimplePreprocessor
from preprocessing import MeanPreprocessor
from io import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator

def load_data(x_train, y_train, x_test, y_test, batch_size):
    """Loads tiny imagenet

    # Returns
        Tuple of HDF% Generators: `(x_train, y_train), (x_test, y_test)`.

    # Raises
        ValueError: in case of invalid `label_mode`.
    """
    # load the RGB means for the training set
    means = json.loads(open(config.DATASET_MEAN).read())

    # initialize the image preprocessors
    sp = SimplePreprocessor(64, 64)
    mp = MeanPreprocessor(means["R"], means["G"], means["B"])
    iap = ImageToArrayPreprocessor()

    # initialize the training and validation dataset generators
    trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, aug=aug,
            preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)
    valGen = HDF5DatasetGenerator(config.VAL_HDF5, 64,
            preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)

    info = "Data generators for hdf5 data and for augmentation"

    return (trainGen, valGen, info)
