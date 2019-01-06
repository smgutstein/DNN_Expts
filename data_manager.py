from __future__ import print_function

import importlib
import numpy as np
import pickle
import types

from data_display import Data_Display

import os
import sys
file_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'dataset_loaders')
if file_dir not in sys.path:
    sys.path.append(file_dir)

class DataManager(object):

    def __init__(self, encoding_activation_fnc,
                 file_param_dict,
                 encoding_param_dict,
                 encoding_module_param_dict,
                 saved_param_dict):

        # Specify number of output nodes in net (i.e. number of bits in encoding)
        self.nb_code_bits = int(encoding_param_dict['nb_code_bits'])
        
        # Init dicts that map class numbers to class names
        self._init_num_name_dicts(file_param_dict['class_names'])

        # Load raw data as numpy arrays
        self.data_loading_module = file_param_dict['data_loader']
        temp = importlib.import_module(self.data_loading_module)
        self._load_data(temp)

        # Import make_encoding_dict method and dynamically make it a member function
        # of this instance of DataManager
        joint_dict = encoding_param_dict.copy()
        joint_dict.update(encoding_module_param_dict)
        joint_dict['encoding_activation_fnc'] = encoding_activation_fnc

        # If recovering saved net, ensure that the encoding used for that net is recovered
        if len(saved_param_dict) > 0:
            self.encoding_module = "recover_encoding"
            joint_dict['saved_encodings'] = \
                os.path.join(saved_param_dict['saved_set_dir'],
                             saved_param_dict['saved_dir'],
                             saved_param_dict['saved_dir'] +
                             '_encodings_' +
                             saved_param_dict['saved_encodings_iter'] +
                             '.pkl')
        else:
            self.encoding_module = encoding_module_param_dict['encoding_module']
            
        temp = importlib.import_module(self.encoding_module)
        self.make_encoding_dict = types.MethodType(temp.make_encoding_dict,
                                                   self)
        self.make_encoding_dict(**joint_dict)
        self.encode_labels()
        self.curr_encoding_info = dict()
        self.curr_encoding_info['label_dict'] = {}
        self.curr_encoding_info['encoding_dict'] = {}
        self.data_display = Data_Display(self.X_test, self.y_test,
                                         self.label_dict)

    def _init_num_name_dicts(self, category_name_file):
        # Make class_num/class_name dictionaries
        with open(category_name_file, "r") as f:
            self.label_dict = pickle.load(f)

    def _load_data(self, data_load_module):
        print("Loading data")
        (self.X_train, self.y_train), \
        (self.X_test, self.y_test) = data_load_module.load_data()
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')

        _, temp1, temp2, temp3 = self.X_train.shape
        if min(temp1, temp2, temp3) == temp3:
            # Data channels last
            _, self.img_rows, self.img_cols, self.img_channels = self.X_train.shape
        elif min(temp1, temp2, temp3) == temp1:
            # Data channels first
            _, self.img_channels, self.img_rows, self.img_cols = self.X_train.shape
                        
        # Rescale raw data
        self.X_train /= 255.
        self.X_test /= 255.
        
        # Save copy of original datasets
        self.X_train_orig = self.X_train
        self.X_test_orig = self.X_test
        self.y_train_orig = self.y_train
        self.y_test_orig = self.y_test

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
        temp = nb_2_encoding_dict.keys()[0]
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
