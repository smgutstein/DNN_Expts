from __future__ import print_function
import importlib
import numpy as np
import os
import pickle
import sys
import types

from keras.preprocessing.image import ImageDataGenerator

ImageDataGen_args = {'featurewise_center': False, 'samplewise_center': False,
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


class DataManager(object):

    def __init__(self, encoding_activation_fnc,
                 file_param_dict,
                 encoding_param_dict,
                 encoding_module_param_dict,
                 saved_param_dict,
                 expt_param_dict,
                 trgt_task_param_dict,
                 preprocess_param_dict,
                 augment_param_dict):        
         
        # Get batch size for data_generator module
        if 'batch_size' in expt_param_dict:
            self.batch_size = int(expt_param_dict['batch_size'])
        else:
            self.batch_size = 32

        ####################################################################################
        fpd = file_param_dict
        tpd = trgt_task_param_dict
        epd = encoding_param_dict
        empd = encoding_module_param_dict
        eaf = encoding_activation_fnc

        # Get path to data
        self.data_path = fpd.get('data_path', None)
        
        # Get data loader
        self.data_loading_module = fpd.get('data_loader', None)
        if self.data_loading_module is None:
            print("Data loading module needs to be specified")
            sys.exit()

        # Determine if data comes from src or trgt task
        if len(trgt_task_param_dict) == 0:            
            # Specify number of output nodes in net (i.e. number of bits in encoding)
            self.nb_code_bits = int(epd['nb_code_bits'])
            self.src_nb_code_bits = int(epd['nb_code_bits'])
            # Init dicts that map class numbers to class names
            self._init_num_name_dicts(fpd['class_names'])

        else:
            # Specify number of output nodes in net (i.e. number of bits in encoding)
            self.nb_code_bits = int(tpd['_EncodingParamDict']['nb_code_bits'])
            self.src_nb_code_bits = int(epd['nb_code_bits'])
            # Init dicts that map class numbers to class names
            self._init_num_name_dicts(tpd['class_names'])

        # Get info to create or recover encoding dict
        if len(trgt_task_param_dict) != 0:            
            epd = tpd['_EncodingParamDict']
            empd = tpd['_EncodingModuleParamDict']
        self.joint_dict = epd.copy()
        self.joint_dict.update(empd)
        self.joint_dict['encoding_activation_fnc'] = eaf

        # Set encoding module
        # If recovering saved net, ensure that the encoding used for that net is recovered
        if len(trgt_task_param_dict) != 0:
            # Encoding for target task
            self.encoding_module = \
                trgt_task_param_dict['_EncodingModuleParamDict']['encoding_module']

        elif len(saved_param_dict) > 0:
            # Recover encoding from saved task
            self.encoding_module = "recover_encoding"
            self.joint_dict['saved_encodings'] = \
                os.path.join(saved_param_dict['saved_set_dir'],
                             saved_param_dict['saved_dir'],
                             'checkpoint_encodings_' +
                             str(int(saved_param_dict['saved_encodings_iter'])) +
                             '.pkl')
        else:
            # Fresh encoding for new src task
            self.encoding_module = empd['encoding_module']
            
        # Load data
        self.ImageDataGen_args = ImageDataGen_args
        self.preprocess_param_dict = preprocess_param_dict
        self.augment_param_dict = augment_param_dict
        self.train_data_gen, self.train_batches_per_epoch = self._load_data('train')
        self.test_data_gen, self.test_batches_per_epoch = self._load_data('test')

        self.curr_encoding_info = dict()
        self.curr_encoding_info['label_dict'] = {}
        self.curr_encoding_info['encoding_dict'] = {}
        #######################################################################################
        
    def _init_num_name_dicts(self, category_name_file):
        # Make class_num/class_name dictionaries
        with open(category_name_file, "rb") as f:
            self.label_dict = pickle.load(f)

    def _load_data(self, trvate='train'):
        # Load data
        data_load_module = importlib.import_module("dataset_loaders." + self.data_loading_module)
        self.load_data = types.MethodType(data_load_module.load_data, self)

        print("Loading data from",
              os.path.join("dataset_loaders",
                           self.data_loading_module))

        # Overwrite default args for ImageDataGenerator with those from cfg files
        # (This might just be a little silly - could just combine dicts created from
        # cfg files and not explicitly state default values, but for now helps me keep
        # track of what default values are)
        for temp in self.preprocess_param_dict:
            self.ImageDataGen_args[temp] = self.preprocess_param_dict[temp]
        for temp in self.augment_param_dict:
            self.ImageDataGen_args[temp] = self.augment_param_dict[temp]

        # Adding encoding
        encoding_module = importlib.import_module(self.encoding_module)
        self.make_encoding_dict = types.MethodType(encoding_module.make_encoding_dict, self)

        data_gen, batches_per_epoch = self.load_data(trvate)
        return data_gen, batches_per_epoch

    def get_encoding_dict(self, class_nums):
        # Get sorted list of class numbers (np.unique returns sorted list)
        self.class_nums = class_nums
        # Note: All below involves getting/making global properties to be
        #       used with data. Current implementation assumes all data can
        #       be read in at once. After creating hdf5 generators, will
        #       see if following code can be moved to other modules, so
        #       that "load_data" will *ONLY* load data.

        self.make_encoding_dict(**self.joint_dict)
        self.encodings = np.asarray([self.encoding_dict[x]
                                     for x in sorted(self.encoding_dict)])

    def get_input_shape(self, x_shape):
        # Get rows, cols and channels. Assume smallest dim, other than 0th
        # is channel dim
        _, temp1, temp2, temp3 = x_shape
        if min(temp1, temp2, temp3) == temp3:
            # Data channels last
            _, self.img_rows, self.img_cols, self.img_channels = x_shape
        elif min(temp1, temp2, temp3) == temp1:
            # Data channels first
            _, self.img_channels, self.img_rows, self.img_cols = x_shape

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

    def get_data_classes_summary_str(self):
        names_per_row = 5
        out_str = "\n\nTotal Classes: " + str(len(self.class_nums)) + "\n"

        for ctr, class_ctr in enumerate(range(len(self.class_nums))):
            if ctr % names_per_row == 0:
                out_str += "\n"
            curr_class = self.class_nums[class_ctr]
            out_str += str(curr_class) +  ":" + self.label_dict[curr_class] + " -- "
        out_str += "\n"
        return out_str


if __name__ == '__main__':
    pass
