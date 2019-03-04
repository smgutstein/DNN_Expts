# import the necessary packages
import os

ROOT_DIR = "/media/smgutstein/current_data/tiny-imagenet-200"

# define the paths to the training and validation directories
TRAIN_IMAGES = os.path.join(ROOT_DIR, "train")
VAL_IMAGES = os.path.join(ROOT_DIR,"val/images")

# define the path to the file that maps validation filenames to
# their corresponding class labels
VAL_MAPPINGS = os.path.join(ROOT_DIR,"val/val_annotations.txt")

# define the paths to the WordNet hierarchy files which are used
# to generate our class labels
WORDNET_IDS = os.path.join(ROOT_DIR,"wnids.txt")
WORD_LABELS = os.path.join(ROOT_DIR,"words.txt")

# since we do not have access to the testing data we need to
# take a number of images from the training data and use it instead
NUM_CLASSES = 200
NUM_TEST_IMAGES = 50 * NUM_CLASSES

# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = os.path.join(ROOT_DIR,"hdf5/train.hdf5")
VAL_HDF5 = os.path.join(ROOT_DIR,"hdf5/val.hdf5")
TEST_HDF5 = os.path.join(ROOT_DIR,"hdf5/test.hdf5")

# define the path to the dataset mean
DATASET_MEAN = os.path.join(ROOT_DIR,"output/tiny-image-net-200-mean.json")

