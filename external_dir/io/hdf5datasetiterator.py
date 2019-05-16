# import the necessary packages
from keras.utils import np_utils
import numpy as np
import h5py


class HDF5DatasetIterator:
    def __init__(self, dbPath, batch_size, nb_code_bits,
                 preprocessors=None, aug=None):
        # store the batch size, preprocessors, and data augmentor,
        # whether or not the labels should be binarized, along with
        # the total number of classes
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.aug = aug
        self.nb_code_bits = nb_code_bits

        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.db = h5py.File(dbPath, 'r')
        self.num_images = self.db['labels'].shape[0]
        self.batches_per_epoch = self.num_images // self.batch_size
        self.curr_batch = 0
        
        # Get sorted list of class numbers (np.unique returns sorted list)
        self.class_nums = sorted(list(set(self.db['labels'])))
        self.encoding_dict = None

        self.curr_epoch = 0

    def set_encoding_dict(self, encoding_dict):
        self.encoding_dict = encoding_dict

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_images

    def __next__(self):
        return self.next()

    def _get_next_batch(self):
        if self.curr_batch == self.batches_per_epoch :
            self.curr_batch = 0
            self.curr_epoch += 1

        offset = self.curr_batch * self.batch_size

        # extract the images and labels from the HDF dataset
        images = self.db["images"][offset: offset + self.batch_size]
        labels = self.db["labels"][offset: offset + self.batch_size]
        self.curr_batch += 1

        return images, labels

    def next(self):

        # Get next batch of raw images and labels
        images, labels = self._get_next_batch()

        # Encode labels
        if self.encoding_dict:
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
            # Note: self.aug.flow returns an iterator
            (images, labels) = next(self.aug.flow(images,
                                                  labels, batch_size=self.batch_size))

        # return a tuple of images and labels
        return (images, labels)

    def close(self):
        # close the database
        self.db.close()
