# import the necessary packages
from keras.utils import np_utils
import h5py
import numpy as np
import random
import sys



class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, nb_code_bits, image_gen):
        # store the batch size, preprocessors, and data augmentor,
        # whether or not the labels should be binarized, along with
        # the total number of classes
        self.batchSize = batchSize
        self.image_gen = image_gen
        self.nb_code_bits = nb_code_bits

        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.db = h5py.File(dbPath, 'r')
        self.numImages = self.db['labels'].shape[0]
        self.imgShape = self.db['images'].shape
        self.batches_per_epoch = self.numImages // self.batchSize + 1

        self.encoding_dict = None

        self.epochs = 0
        self.passes = np.inf

    def set_encoding_dict(self, encoding_dict):
        self.encoding_dict = encoding_dict

    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0

        #all_images = self.db["images"]
        #all_labels = self.db["labels"]
        #self.image_gen.fit(all_images)

        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:

            # loop over the HDF5 dataset
            batch_list = np.arange(0, self.numImages, self.batchSize)
            random.shuffle(batch_list)
            for i in batch_list:
                # extract the images and labels from the HDF dataset
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]

                self.image_gen.fit(images)
                # Encode class
                encoded_labels = np.empty((len(labels), self.nb_code_bits))
                for i in range(len(labels)):
                    encoded_labels[i, :] = self.encoding_dict[labels[i]]
                labels = encoded_labels

                (images, labels) = next(self.image_gen.flow(images, labels,
                                                            batch_size=self.batchSize))

                # yield a tuple of images and labels
                yield images, labels

            # increment the total number of epochs
            epochs += 1

    def close(self):
        # close the database
        self.db.close()
