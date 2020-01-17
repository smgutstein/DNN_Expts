import argparse
import os

head_str = '''
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


def load_data(label_mode='fine'):
    \"\"\"Loads trgt tasks for CIFAR100 living_vs_notliving datasets.

    # Arguments
        label_mode: one of \"fine\", \"coarse\".

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    # Raises
        ValueError: in case of invalid `label_mode`.
    \"\"\"
    if label_mode not in [\'fine\', \'coarse\']:
        raise ValueError(\'`label_mode` must be one of `\"fine\"`, `\"coarse\"`.')

    path = os.path.join(home,\'.keras/datasets/\', \'cifar-100-python\',
                        \'Living_vs_Not_Living\', \'trgt_tasks_\' '''

tail_str = ''' ) 

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
    ap = argparse.ArgumentParser()
    ap.add_argument("suffix", type = str,
                    help="samples per class")
    args = ap.parse_args()
    suffix = args.suffix 
    outstr = head_str + " + \'" + args.suffix +"\'" + tail_str
    outpath  = './dataset_loaders/'
    outfile = 'cifar100_trgt_living_vs_notliving_subset_' + suffix + '.py'
    with open(os.path.join(outpath,outfile), 'w') as f:
        f.write(outstr)
    
