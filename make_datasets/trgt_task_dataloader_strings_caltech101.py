import_str = '''
from __future__ import absolute_import
import os
import sys
file_dir = os.path.dirname(os.path.realpath(__file__))
if file_dir not in sys.path:
    sys.path.append(file_dir)
from caltech101 import load_batch
from keras.utils.data_utils import get_file
from keras import backend as K
import numpy as np
from os.path import expanduser
home = expanduser("~")

'''

header_str = '''
def load_data(self, trvate='train', label_mode='fine'):
'''

doc_body_str = '''
    
    
    # Arguments
        trvate: one of 'train' or 'test'
        label_mode: one of \"fine\", \"coarse\".

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    # Raises
        ValueError: in case of invalid `label_mode`.

'''

path_str = '''
    path = os.path.join(home, self.data_path)
    fpath = os.path.join(path, trvate)\n\n'''

body_str = '''
    if label_mode not in [\'fine\', \'coarse\']:
        raise ValueError(\'`label_mode` must be one of `\"fine\"`, `\"coarse\"`.')


    #Get data
    x_data, y_data = load_batch(fpath, label_key=label_mode + '_labels')
    y_data = np.reshape(y_data, (len(y_data), 1))

    # Rescale raw data
    x_data = x_data.astype('float32')
    x_data /= 255.

    if K.image_data_format() == 'channels_last':
        x_data = x_data.transpose(0, 2, 3, 1)

    # Get sorted list of class numbers (np.unique returns sorted list)
    class_nums = sorted(list(np.unique(y_data)))

    # Get shape of input images
    self.get_input_shape(x_data.shape)

    # Adding encoding
    self.get_encoding_dict(class_nums)

    nb_outputs = self.encodings.shape[1]

    y_temp = y_data.ravel()
    y_data_encoded = np.empty((len(y_temp), nb_outputs))
    for i in range(len(y_temp)):
        y_data_encoded[i, :] = self.encoding_dict[y_temp[i]]

    # Set batches (i.e. steps) per epoch
    batches_per_epoch = y_data_encoded.shape[0] // self.batch_size + 1

    print("Preprocessing " + trvate + " Images")
    image_gen = ImageDataGenerator(**self.ImageDataGen_args)
    image_gen.fit(x_data)
    data_gen = image_gen.flow(x_data, y_data_encoded, self.batch_size)

    return data_gen, batches_per_epoch

'''
