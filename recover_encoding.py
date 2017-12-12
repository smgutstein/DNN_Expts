from __future__ import print_function
from datetime import datetime
import hot_not_hot_values as hnh
import numpy as np
import pickle
from random import random


def make_encoding_dict(self, **kwargs):

    encoding_file = kwargs['saved_encodings']
    print("Recovering Saved Encodings from %s" % encoding_file)
    pickled_dict = pickle.load(open(encoding_file,'r'))

    self.encoding_dict = pickled_dict['encoding_dict']
    self.label_dict = pickled_dict['label_dict']
    self.meta_encoding_dict = pickled_dict['meta_encoding_dict']
    self.not_hot = self.meta_encoding_dict['not_hot']
    self.hot = self.meta_encoding_dict['hot']
    self.encoding_type = self.meta_encoding_dict['encoding_type']

    class_nums = len(self.encoding_dict)
    temp = self.encoding_dict[self.encoding_dict.keys()[0]]
    nb_code_bits = temp.shape[0]
    
    self.encoding_matrix = np.zeros((class_nums, nb_code_bits))
    for ctr, curr_class_num in enumerate(sorted(self.encoding_dict)):
        self.encoding_matrix[ctr, :] = self.encoding_dict[curr_class_num] 
 
