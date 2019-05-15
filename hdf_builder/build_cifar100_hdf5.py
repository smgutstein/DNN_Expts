# USAGE
# python build_tiny_imagenet.py

# import the necessary packages
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cifar import load_batch
from config import cifar100 as config
import cv2
from external_dir.utils import HDF5DatasetWriter
from external_dir.imutils import paths
import json
from keras import backend as K
from keras.utils.data_utils import get_file
import numpy as np
import progressbar
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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



# construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output HDF5
# files
(x_train, y_train), (x_test, y_test) = load_data()
datasets = [
	("train", x_train, y_train, config.TRAIN_HDF5),
	("test", x_test, y_test, config.TEST_HDF5)]

# initialize the lists of RGB channel averages
(R, G, B) = ([], [], [])

# loop over the dataset tuples
for (dType, data, labels, outputPath) in datasets:
	# create HDF5 writer
	print("[INFO] building {}...".format(outputPath))
	writer = HDF5DatasetWriter(data.shape, outputPath)

	# initialize the progress bar
	widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
		progressbar.Bar(), " ", progressbar.ETA()]
	pbar = progressbar.ProgressBar(maxval=data.shape[0],
		widgets=widgets).start()

        for (i, (datum, label)) in enumerate(zip(data, labels)):
            # if we are building the training dataset, then compute the
            # mean of each channel in the image, then update the
            # respective lists
            if dType == "train":
                (r, g, b) = np.mean(np.mean(datum,0),0)
                R.append(r)
                G.append(g)
                B.append(b)

            # add the image and label to the HDF5 dataset
            writer.add([datum], [label[0]])
            pbar.update(i)

	# close the HDF5 writer
	pbar.finish()
	writer.close()

# construct a dictionary of averages, then serialize the means to a
# JSON file
print("[INFO] serializing means...")
D = {"R": float(np.mean(R)), "G": float(np.mean(G)), "B": float(np.mean(B))}
print D

f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()
