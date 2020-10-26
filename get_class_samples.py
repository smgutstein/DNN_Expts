from __future__ import print_function
from collections import defaultdict
import configparser
import importlib
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import sys
import types


def is_number(in_str):
    try:
        float(in_str)
        return True
    except ValueError:
        return False

ImageDataGen_args = {'featurewise_center': False,
                     'samplewise_center': False,
                     'featurewise_std_normalization': False,
                     'samplewise_std_normalization': False,
                     'zca_epsilon': 1e-6, 'zca_whitening': False,
                     'rotation_range': 0,
                     'width_shift_range': 0, 'height_shift_range': 0,
                     'brightness_range': None,
                     'shear_range': 0.0, 'zoom_range': 0.0,
                     'channel_shift_range': 0.0,
                     'fill_mode': 'nearest',
                     'cval': 0,
                     'horizontal_flip': False, 'vertical_flip': False,
                     'rescale': None,
                     'preprocessing_function': None,
                     'validation_split': 0,
                     'dtype': 'float32'}


class SamplesCollater(object):
    '''Supplies data generator for either train or test set
       used in previous expt.'''

    def __init__(self, forensics_file):
        # Get info on expt & data to be examined
        self.config = configparser.ConfigParser()
        self.forensics_file = forensics_file
        self.quiet = False
        self.class_nums = set()
        self.recover_expt_params()

        # Get data
        self.get_raw_data()
        self.encode_labels()

        # Get train & test sets
        self.get_train_test_sets()

        # Create dataset generator - assuming for training set
        self.curr_set = None
        self.get_full_generators()

    # Two silly little routines to control
    # information printed when calling choose_dataset
    def make_quiet(self):
        self.quiet = True
    def make_verbose(self):
        self.quiet = False

    def get_param_dict(self, dict_name):

        # Convert info from configparser to standard dict
        param_dict = {}
        try:
            params = self.config.items(dict_name)
            for curr_pair in params:
                param_dict[curr_pair[0]] = curr_pair[1]

            # Convert numeric strings to float
            param_dict = {xx: float(param_dict[xx])
                          if is_number(param_dict[xx])
                          else param_dict[xx]
                          for xx in param_dict}

            # Convert non-numeric strings to correct variable types
            for x in param_dict:
                if str(param_dict[x]).lower() == 'none':
                    param_dict[x] = None
                elif str(param_dict[x]).lower() == 'true':
                    param_dict[x] = True
                elif str(param_dict[x]).lower() == 'false':
                    param_dict[x] = False

        except configparser.NoSectionError:
            pass
        return param_dict

    def recover_expt_params(self):
        
        if not os.path.isfile(self.forensics_file):
            print("Can't find %s. Is it a file?" % self.forensics_file)
            sys.exit()
        self.config.read(self.forensics_file)

        # Get path to cfg file for orig. expt
        path_param_dict = self.get_param_dict('SavedNetPathParams')
        expt_cfg_file = os.path.join(path_param_dict['root_dir'],
                                     path_param_dict['expt_dir'],
                                     path_param_dict['arch_dir'],
                                     path_param_dict['net_type'],
                                     path_param_dict['cfg_file'])
        cluster_param_dict = self.get_param_dict('RawDataPathParams')

        # Get cfgs for orig. expt
        if not os.path.isfile(expt_cfg_file):
            print("Can't find %s. Is it a file?" % expt_cfg_file)
            sys.exit()
        self.config.read(expt_cfg_file)

        # Get Data Loader
        file_param_dict = self.get_param_dict('ExptFiles')
        self.data_loading_module_name = \
            cluster_param_dict.get('data_loader',
                                   file_param_dict.get('data_loader', None))

        # Get preprocessing data
        net_param_dict = self.get_param_dict('NetParams')
        preprocess_param_dict = \
            self.get_param_dict('DataPreprocessParams')
        
        # Construct dict used to recover class encoding.
        # Copying code structure from data_manager.py
        if not os.path.isfile(file_param_dict['encoding_cfg']):
            print("Can't find %s. Is it a file?" % expt_cfg_file)
            sys.exit()
        self.config.read(file_param_dict['encoding_cfg'])

        # Recover data encoding
        encoding_param_dict = self.get_param_dict('Encoding')
        self.nb_code_bits = int(encoding_param_dict['nb_code_bits'])

        encoding_module_param_dict = \
            self.get_param_dict('EncodingModuleParams')
        self.encoding_module = \
            encoding_module_param_dict['encoding_module']
        self.joint_dict = encoding_param_dict.copy()
        self.joint_dict.update(encoding_module_param_dict)
        self.joint_dict['encoding_activation_fnc'] = \
            net_param_dict['output_activation']

        if 'trgt_set' not in cluster_param_dict or \
            cluster_param_dict['trgt_set'] == 'src_tasks':
            self.encoding_module = 'recover_encoding'
            self.joint_dict['saved_encodings'] = \
                os.path.join(file_param_dict['root_expt_dir'],
                             path_param_dict['expt_dir'],
                             path_param_dict['arch_dir'],
                             path_param_dict['net_type'],
                             path_param_dict['expt_subdir'],
                             path_param_dict['machine_name'],
                             'checkpoints',
                             'checkpoint_encodings_' + '0' + '.pkl')

        # Get preprocessing params
        for x in preprocess_param_dict:
            ImageDataGen_args[x] = preprocess_param_dict[x]
        self.image_gen = ImageDataGenerator(**ImageDataGen_args)

        # Get batch_size
        expt_param_dict = self.get_param_dict('ExptParams')
        self.batch_size = int(expt_param_dict['batch_size'])

        return None

    def get_raw_data(self):
        '''Get raw data using data loader'''
        # Get data loader
        if self.data_loading_module_name is None:
            print("Data loading module needs to be specified")
            sys.exit()
        mod_str = "dataset_loaders." +  self.data_loading_module_name
        self.data_load_module = \
            importlib.import_module(mod_str)


        # Get data
        print("Loading data from",
              os.path.join("dataset_loaders",
                           self.data_loading_module_name))
        (self.X_train, self.y_train_classnum), \
        (self.X_test, self.y_test_classnum) = \
            self.data_load_module.load_data()
        temp_set = set([x[0] for x in self.y_train_classnum])
        self.class_nums = self.class_nums.union(temp_set)

        return None

    def encode_labels(self):
        """Convert array of class nums to arrays of encodings"""

        # Recover encodings
        print("Recovering Encodings")
        temp = importlib.import_module(self.encoding_module)
        self.make_encoding_dict = \
            types.MethodType(temp.make_encoding_dict, self)
        self.make_encoding_dict(**self.joint_dict)

        # Make array of encodings
        print("Creating encoding matrices for train & test data")
        self.encodings = np.asarray([self.encoding_dict[x]
                                     for x in sorted(self.encoding_dict)])
        
        # Convert labels from class nums to class encodings
        self.Y_train_encoded = \
            self.convert_class_num_to_encoding(self.y_train_classnum,
                                               self.encoding_dict)
        self.Y_test_encoded = \
            self.convert_class_num_to_encoding(self.y_test_classnum,
                                               self.encoding_dict)

        return None

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

    def get_train_test_sets(self):
        '''Create dict mapping raw class nums to
           list of data rows storing samples of that class'''
        # Init default dicts
        self.data_dict_train = defaultdict(list)
        self.data_dict_test = defaultdict(list)

        # Divide training set into class specific sets
        for x in range(self.y_train_classnum.shape[0]):
            self.data_dict_train[int(self.y_train_classnum[x])].append(x)
            
        # Divide testing set into class specific sets
        for x in range(self.y_test_classnum.shape[0]):
            self.data_dict_test[int(self.y_test_classnum[x])].append(x)
        return None

    def choose_dataset(self, train=True):

        prefix_str = "Information for the "
        suffix_str = " dataset is already loaded"
        if train:
            if self.curr_set == 'train':
                if not self.quiet:
                    print(prefix_str +  "training" + suffix_str)
                return None
            else:
                print("Loading information for training dataset")
        else:
            if self.curr_set == 'test':
                if not self.quiet:
                    print(prefix_str +  "testing" + suffix_str)
                return None
            else:
                print("Loading information for testing dataset")

        # Determine if class specific set is to be extracted
        # from training or test set
        if train:
            self.data = self.X_train
            self.data_dict = self.data_dict_train
            self.labels = self.y_train_classnum
            self.encoded = self.Y_train_encoded
            self.curr_set = 'train'
        else:
            self.data = self.X_test
            self.data_dict = self.data_dict_test
            self.labels = self.y_test_classnum
            self.encoded = self.Y_test_encoded
            self.curr_set = 'test'

        return None

    def get_class_data(self, curr_class):
        '''Pick out all samples of a given class from either the train or test set
           and create mini-dataset'''
        # Create and initialize output arrays for specific class
        # Ensure shape & dtypes for trgt array & src array match
        num_elems = len(self.data_dict[curr_class])
        code_length = self.encoded.shape[1]
        ds = self.data.shape

        sub_array = np.zeros((num_elems, ds[1], ds[2], ds[3]))
        sub_array = sub_array.astype(self.data.dtype)
        sub_labels = np.zeros((num_elems, 1)) 
        sub_labels = sub_labels.astype(self.labels.dtype)
        sub_encoded = np.zeros((num_elems, code_length))
        sub_encoded = sub_encoded.astype(self.encoded.dtype)
        
        for ctr,row in enumerate(self.data_dict[curr_class]):
            sub_array[ctr,:,:,:] = self.data[row,:,:,:]
            sub_labels[ctr, 0] = self.labels[row,0]
            sub_encoded[ctr, :] = self.encoded[row,:]
        return sub_array, sub_labels, sub_encoded

    def get_class_generator(self, curr_class, train=True):
        '''Create a data generator for a specific class using either
        the train or test set'''
        # Create Keras image generator for given class
        self.choose_dataset(train)
        self.sub_data, self.sub_labels, self.sub_encoded = \
            self.get_class_data(curr_class)
        self.image_gen.fit(self.sub_data)
        self.sub_data_gen = self.image_gen.flow(self.sub_data,
                                                self.sub_encoded,
                                                self.batch_size)

    def get_full_generators(self, train = True):
        '''Create a data generator using either the train or test set'''

        # Create Keras Image Generator for full dataset
        self.choose_dataset(train)
        self.image_gen.fit(self.data)
        self.data_gen = self.image_gen.flow(self.data,
                                            self.encoded,
                                            self.batch_size)

        return None


if __name__ == '__main__':

    x = SamplesCollater(os.path.join('./cfg_dir/net_forensics_cfg',
                                     'src_net_0a.cfg'))
