# import the necessary packages
from keras.utils import np_utils
import numpy as np
import h5py


class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, nb_code_bits,
                 preprocessors=None, aug=None):
        # store the batch size, preprocessors, and data augmentor,
        # whether or not the labels should be binarized, along with
        # the total number of classes
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.nb_code_bits = nb_code_bits

        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.db = h5py.File(dbPath, 'r')
        self.numImages = self.db['labels'].shape[0]
        self.batches_per_epoch = self.numImages // self.batchSize
        
        # Get sorted list of class numbers (np.unique returns sorted list)
        self.class_nums = sorted(list(set(self.db['labels'])))
        self.encoding_dict = None

        self.epochs = 0
        self.passes = np.inf

    def set_encoding_dict(self, encoding_dict):
        self.encoding_dict = encoding_dict

    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0
        
        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:
            # loop over the HDF5 dataset
            for i in np.arange(0, self.numImages, self.batchSize):
                # extract the images and labels from the HDF dataset
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]

                # Encode class
                encoded_labels = np.empty((len(labels), self.nb_code_bits))
                for i in range(len(labels)):
                    encoded_labels[i, :] = self.encoding_dict[labels[i]]
                labels = encoded_labels

                # check to see if our preprocessors are not None
                if self.preprocessors is not None:
                    # initialize the list of processed images
                    procImages = []

                    # loop over the images
                    for image in images:
                        # loop over the preprocessors and apply each
                        # to the image
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        # update the list of processed images
                        procImages.append(image)

                    # update the images array to be the processed
                    # images
                    images = np.array(procImages)

                # if the data augmenator exists, apply it
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images,
                    labels, batch_size=self.batchSize))

                # yield a tuple of images and labels
                yield (images, labels)

            # increment the total number of epochs
            epochs += 1

    def close(self):
        # close the database
        self.db.close()
