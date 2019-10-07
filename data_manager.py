from __future__ import print_function

import importlib
import numpy as np
import pickle
import types

from data_display import Data_Display

import os
import sys

class DataManager(object):

    def __init__(self, encoding_activation_fnc,
                 file_param_dict,
                 encoding_param_dict,
                 encoding_module_param_dict,
                 saved_param_dict,
                 expt_param_dict):

        # Specify number of output nodes in net (i.e. number of bits in encoding)
        self.nb_code_bits = int(encoding_param_dict['nb_code_bits'])
        
        # Init dicts that map class numbers to class names
        self._init_num_name_dicts(file_param_dict['class_names'])

        # Get info to load data 
        self.data_loading_module = file_param_dict.get('data_loader', None)
        self.data_generator_module = file_param_dict.get('data_generator', None)

        # Get info to create or recover encoding dict
        joint_dict = encoding_param_dict.copy()
        joint_dict.update(encoding_module_param_dict)
        joint_dict['encoding_activation_fnc'] = encoding_activation_fnc

        # If recovering saved net, ensure that the encoding used for that net is recovered
        if len(saved_param_dict) > 0:
            self.encoding_module = "recover_encoding"
            joint_dict['saved_encodings'] = \
                os.path.join(saved_param_dict['saved_set_dir'],
                             saved_param_dict['saved_dir'],
                             'checkpoint_encodings_' +
                             saved_param_dict['saved_encodings_iter'] +
                             '.pkl')
        else:
            self.encoding_module = encoding_module_param_dict['encoding_module']


        # Get batch size for data_gnerator module
        if 'batch_size' in expt_param_dict:
            self.batch_size = int(expt_param_dict['batch_size'])
        else:
            self.batch_size = 32

        # Create encodings
        temp = importlib.import_module(self.encoding_module)
        self.make_encoding_dict = types.MethodType(temp.make_encoding_dict, self)
        

        if self.data_generator_module:
            temp = importlib.import_module("dataset_loaders." + self.data_generator_module)
            self.get_generator = types.MethodType(temp.get_generator, self)

        # Load raw data and/or data generator
        if self.data_loading_module is None and self.data_generator_module is None:
            print("Either or both a data loading module and data generator module must be specified")
            
        elif self.data_loading_module:
            print("Loading data into memory")
            self._load_data()
            self.batches_per_epoch = self.X_train.shape[0] // self.batch_size

            # Encode labels
            print("Encoding data")
            self.make_encoding_dict(**joint_dict)
            self.encode_labels()

            # Might not need/want this anymore
            self.data_display = Data_Display(self.X_test, self.y_test,
                                             self.label_dict)
            # Load data generator - if necessary for augmentation
            if self.data_generator_module:
                print ("Loading data generator for augmentation")
                (self.train_data_generator,
                 self.test_data_generator,
                 self.data_generator_info) = self.get_generator()
                
            else:
                print ("No data augmentation")
                self.train_data_generator = None
                self.test_data_generator = None
                self.data_generator_info = "No data generator being used"

        else:
            print ("Loading data generator")
            (self.train_data_generator,
             self.test_data_generator,
             self.data_generator_info,
             (self.img_channels,
              self.img_rows,
              self.img_cols)) = self.get_generator()

            # Get sorted list of class numbers (np.unique returns sorted list)
            self.class_nums = self.test_data_generator.class_nums
            self.batches_per_epoch = self.train_data_generator.batches_per_epoch

            # Encode labels
            print("Making Encoding Dict")
            self.make_encoding_dict(**joint_dict)
            self.train_data_generator.set_encoding_dict(self.encoding_dict)
            self.test_data_generator.set_encoding_dict(self.encoding_dict)



        self.curr_encoding_info = dict()
        self.curr_encoding_info['label_dict'] = {}
        self.curr_encoding_info['encoding_dict'] = {}

        #self._make_data_generator()

    def _init_num_name_dicts(self, category_name_file):
        # Make class_num/class_name dictionaries
        with open(category_name_file, "rb") as f:
            self.label_dict = pickle.load(f)

    def _load_data(self):
        # Load data
        data_load_module = importlib.import_module("dataset_loaders." + self.data_loading_module)
        print("Loading data")
        (self.X_train, self.y_train), \
        (self.X_test, self.y_test) = data_load_module.load_data()
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')

        # Get rows, cols and channels. Assume smallest dim, other than 0th
        # is channel dim
        _, temp1, temp2, temp3 = self.X_train.shape
        if min(temp1, temp2, temp3) == temp3:
            # Data channels last
            _, self.img_rows, self.img_cols, self.img_channels = self.X_train.shape
        elif min(temp1, temp2, temp3) == temp1:
            # Data channels first
            _, self.img_channels, self.img_rows, self.img_cols = self.X_train.shape

        # Get sorted list of class numbers (np.unique returns sorted list)
        self.class_nums = list(np.unique(self.y_train))


    def encode_labels(self):
        """Convert array of class nums to arrays of encodings"""

        print("Creating encoding matrices for train & test data")      
        # Make array of encodings
        self.encodings = np.asarray([self.encoding_dict[x]
                                     for x in sorted(self.encoding_dict)])
        
        # Convert labels from class nums to class encodings
        self.Y_train = self.convert_class_num_to_encoding(self.y_train,
                                                          self.encoding_dict)
        self.Y_test = self.convert_class_num_to_encoding(self.y_test,
                                                         self.encoding_dict)

    def convert_class_num_to_encoding(self, y, nb_2_encoding_dict):
        """Take vector of class numbers and convert to array
           of target output codes for each class"""

        # Find number of output bits by looking at arbitrary
        # code word
        temp = sorted(list(nb_2_encoding_dict.keys()))[0]
        nb_outputs = len(nb_2_encoding_dict[temp])

        # Turn 2D array into 1D list of class numbers    
        y = y.ravel()
        Y = np.empty((len(y), nb_outputs))

        # Create encoding matrix - ith row is
        # code word for ith datum
        for i in range(len(y)):
            Y[i, :] = nb_2_encoding_dict[y[i]]
        return Y

    def display(self):
            
        self.data_display.start_display()

    def get_targets_str(self):
        nums_per_row = 20
        digits_per_num = 5
        fmt_str = '{: ' + str(digits_per_num) + '.4f}'
        out_str = '\n'

        for curr_class, curr_targets in sorted(self.encoding_dict.items()):
            indent = len(str(curr_class) + ': ') 
            indent_str1 = ' ' * indent
            curr_line = str(curr_class) + ': '
            spacer = ' ' * 2
            formatted_targets = [fmt_str.format(x) for x in curr_targets]
            line_len = 0
            for x in formatted_targets:
                if line_len >= nums_per_row:
                    curr_line += '\n' + indent_str1
                    line_len = 0
                curr_line += x + spacer
                line_len += 1
            out_str += curr_line + '\n'
        return out_str + '\n\n'

    def get_targets_str_sign(self):
        nums_per_row = 20
        out_str = '\n'

        for curr_class, curr_targets in sorted(self.encoding_dict.items()):
            indent = len(str(curr_class) + ': ')
            indent_str1 = ' ' * indent
            curr_line = str(curr_class) + ': '
            spacer = ' ' * 2

            formatted_targets = ['+' if x == self.hot else '-'
                                 for x in curr_targets]
            line_len = 0
            for x in formatted_targets:
                if line_len >= nums_per_row:
                    curr_line += '\n' + indent_str1
                    line_len = 0
                curr_line += x + spacer
                line_len += 1
            out_str += curr_line + '\n'
        return out_str + '\n\n'


if __name__ == '__main__':
    pass
