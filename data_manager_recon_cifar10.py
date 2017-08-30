from __future__ import print_function

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.utils import encoding_utils as enc_utils
from keras import backend as K

import copy
from collections import defaultdict
import json
import itertools
import numpy as np
import pickle
import random
import sys

from display_cifar10 import Data_Display
#from encoding import Encoding

class DataManager(object):

    def __init__(self, nb_code_bits=10):

        #Specify number of output nodes in net (i.e. number of bits in encoding)
        self.nb_code_bits = nb_code_bits
        
        #Init dicts that map class numbers to class names
        self._init_num_name_dicts()

        #Load raw data as numpy arrays
        self._load_data()

        #Initialize TBD attributes 
        self.encoding_dict = None

    def _init_num_name_dicts(self):       
        # Make class_num/class_name dictionaries
        with open("cifar10_dicts_all.pkl","r") as f:
             self.label_dict = pickle.load(f)

    def _load_data(self):
        #Load data
        print("Loading data")
        (self.X_train, self.y_train), \
        (self.X_test, self.y_test) = cifar10.load_data()
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
                        
        # Rescale raw data
        self.X_train /= 255.
        self.X_test /= 255.
        
        # Save copy of original datasets
        self.X_train_orig = self.X_train
        self.X_test_orig = self.X_test
        self.y_train_orig = self.y_train
        self.y_test_orig = self.y_test

    def make_n_hot_encoding_dict(self, hot=1.0, not_hot=0.0, nb_hot=1):
        '''Create n-hot codes'''
        print("Creating %d-hot encodings"%(nb_hot))      
        # relu hot/not_hot
        self.not_hot = not_hot
        self.hot = hot

        #Make n-hot encoding dict
        self.encoding_dict = \
          enc_utils.make_n_hot_nb_2_encoding_dict(self.y_train, self.nb_code_bits,
                                                     not_hot, hot, nb_hot)
        #self._record_encoding()
        self.encoding_type = str(nb_hot) + "-hot"

    def encode_labels(self):
        '''Convert array of class nums to arrays of encodings'''

        print("Creating encoding matrices for train & test data")      
        #Make array of encodings
        self.encodings = np.asarray([self.encoding_dict[x] for x in self.encoding_dict])
        
        # Convert labels from class nums to class encodings
        self.Y_train = enc_utils.make_encoding_matrix(self.y_train, self.encoding_dict)
        self.Y_test = enc_utils.make_encoding_matrix(self.y_test, self.encoding_dict)

    def display(self):
            
        Data_Display(self.X_test, self.y_test, self.label_dict)

    def get_targets_str(self):
        indent_len = 8
        bits_per_row = bpr = 9
        out_str = '\n'
        for curr_class, curr_target in sorted(self.encoding_dict.items()):
            curr_target_str = ''
            out_str += str(curr_class) + ': '
            curr_target_strs = [str(curr_target[x:x+bits_per_row])+'\n'
                                if x+bits_per_row < len(curr_target)
                                else str(curr_target[x:])+'\n'
                                for x in range(0,len(curr_target+1),bpr)]
            indent = max(1,indent_len - len(str(curr_class) + ': '))
            indent_str1 = ' ' * indent
            indent_str2 = ' ' * indent_len

            curr_target_str = indent_str2.join(curr_target_strs)
            curr_target_str = indent_str1 + curr_target_str
            out_str += curr_target_str
        return out_str + '\n\n'

if __name__ == '__main__':
    pass
    
