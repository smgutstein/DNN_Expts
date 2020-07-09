import argparse
import configparser
import itertools
import os

import_str = '''
from __future__ import absolute_import
import os
import sys
file_dir = os.path.dirname(os.path.realpath(__file__))
if file_dir not in sys.path:
    sys.path.append(file_dir)
from cifar import load_batch
from keras.utils.data_utils import get_file
from keras import backend as K
import numpy as np
from os.path import expanduser
home = expanduser("~")

'''

header_str = '''
def load_data(label_mode='fine'):
'''

doc_str_body = '''
    
    
    # Arguments
        label_mode: one of \"fine\", \"coarse\".

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    # Raises
        ValueError: in case of invalid `label_mode`.

'''

path_str = '''
    path = os.path.join(home'''

body_str = '''
    if label_mode not in [\'fine\', \'coarse\']:
        raise ValueError(\'`label_mode` must be one of `\"fine\"`, `\"coarse\"`.')

    fpath = os.path.join(path, \'train\')
    x_train, y_train = load_batch(fpath, label_key=label_mode + \'_labels\')

    fpath = os.path.join(path, 'test')
    x_test, y_test = load_batch(fpath, label_key=label_mode + \'_labels\')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # Rescale raw data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.
    x_test /= 255.

    if K.image_data_format() == \'channels_last\':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)
'''

if __name__ == '__main__':
    # Get desired samples per class
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--cfg_root", type = str,
                    default = "../cfg_dir/gen_cfg/opt_tfer_expts",
                    help = "root dir for config files")
    ap.add_argument("-s", "--cfg_sub", type = str,
                    default = "cifar_100_living_living_expts",
                    help = "dir for config files for set of expts")
    ap.add_argument("-l", "--cfg_leaf", type = str,
                    default = "tfer_datasets/subsets.cfg",
                    help = "dir for config files for set of expts")
    args = ap.parse_args()

    # Make target dirs and copy info from src dir
    config_file = os.path.join(args.cfg_root,
                              args.cfg_sub, 
                              args.cfg_leaf)
    print("Reading ", config_file)
    config = configparser.ConfigParser()
    config.read(config_file)

    note = "Loads data for " + config['Notes']['note']
    doc_str = '    \"\"\"\n' + '    ' + note + doc_str_body + '    \"\"\"\n'

    data_root_dir = config['StorageDirectory']['data_root_dir']
    data_dir = config['StorageDirectory']['data_dir']
    subset_root_dir = config['StorageDirectory']['subset_root_dir']
    subset_dir = config['StorageDirectory']['subset_dir']
    path_str = ', '.join([path_str,
                          "'" + data_root_dir + "'",
                          "'" + data_dir + "'",
                          "'" + subset_root_dir + "'",
                          "'" + subset_dir])

    # Get spc and training set id suffix for each data loader
    spc_list = [x.strip() for x in config['Subsets']['spc'].split(',')]
    suffix_list = [x.strip() for x in config['Subsets']['suffixes'].split(',')]

    for spc,suffix in itertools.product(spc_list, suffix_list):
        data_path = "_".join([path_str, str(spc), suffix + "'"])
        data_path += ")"

        out_str = import_str + header_str + doc_str + data_path + body_str
        out_path = '../dataset_loaders/'
        out_file = '_'.join([subset_root_dir,subset_dir,spc,suffix,"TEST"]) + '.py'
        
        with open(os.path.join(out_path,out_file), 'w') as f:
            f.write(out_str)
            print("Wrote: ",os.path.join(out_path,out_file))

 
