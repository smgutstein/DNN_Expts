# import the necessary packages
import os

ROOT_DIR = os.path.join(os.environ["HOME"],".keras/datasets/cifar-100-python")

# define the paths to the training and test directories
TRAIN_IMAGES = os.path.join(ROOT_DIR, "train")
TEST_IMAGES = os.path.join(ROOT_DIR,"test")


# since we do not have access to the testing data we need to
# take a number of images from the training data and use it instead
NUM_CLASSES = 100
NUM_TEST_IMAGES = 10000

# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = os.path.join(ROOT_DIR, "hdf5/train.hdf5")
TEST_HDF5 = os.path.join(ROOT_DIR, "hdf5/test.hdf5")

# define the path to the dataset mean
DATASET_MEAN = os.path.join(ROOT_DIR, "hdf5/cifar100-mean.json")

# define the path to the sorted list of class numbers
CLASS_NUMS = os.path.join(ROOT_DIR, "hdf5/class_nums.pkl")

