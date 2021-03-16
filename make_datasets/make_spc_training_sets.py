from __future__ import print_function
import argparse
from collections import defaultdict, Counter
import configparser
import itertools
import numpy as np
import pickle
import os
from os.path import expanduser
import shutil


def convert(data):
    # Converts strings of form b'my_string' to 'my_string'
    # (i.e. converts bytes to str)
    if isinstance(data, bytes): return data.decode()
    if isinstance(data, dict): return dict(map(convert, data.items()))
    if isinstance(data, tuple): return tuple(map(convert, data))
    if isinstance(data, list): return list(map(convert, data))
    return data


def get_subset(train, samps_per_class):
    """Creates subset of a tiny imagenet 200 training set. 
       New subset will have specified
       number of samples per class"""

    num_classes = len(set(train['fine_labels']))
    
    # Initialze info for subset_dict
    subset_data = np.zeros((samps_per_class*num_classes, 3072),dtype=train['data'].dtype)  # 32*32*3=3072
    subset_dict = dict()
    subset_dict['fine_labels'] = []
    subset_dict['coarse_labels'] = []
    subset_dict['filenames'] = [] 
    subset_dict['batch_label'] = "Subset training batch 1 of 1 - " 
    subset_dict['batch_label'] += str(samps_per_class*num_classes) + " samps per class"
    
    # Initialize dict to track number of samples used per class
    used_dict = defaultdict(int)
    
    # Init vars to track how many samples have been gathered 
    # and which element from train dict is about to be considered for the subset
    tot_used = 0

    # Randomize image selection
    candidate_list = list(np.random.permutation(len(train['fine_labels'])))
    curr_candidate = candidate_list.pop()
    
    # Loop until have required samples per class for each class
    while tot_used < samps_per_class*num_classes:
        
        # Get class of next element to be considered and ensure we still want more 
        # samples of that class
        curr_candidate_class = train['fine_labels'][curr_candidate]
        if used_dict[curr_candidate_class] < samps_per_class:
            # Copy chosen sample
            subset_dict['fine_labels'].append(train['fine_labels'][curr_candidate])
            subset_dict['coarse_labels'].append(train['coarse_labels'][curr_candidate])
            subset_dict['filenames'].append(train['filenames'][curr_candidate])
            subset_data[tot_used, :] = train['data'][curr_candidate, :]
            
            # Update tracking variables
            tot_used += 1
            used_dict[curr_candidate_class] += 1
        else:
            pass   
        # Proceed to next candidate element
        curr_candidate = candidate_list.pop()
        
    subset_dict['data'] = subset_data
    print("tot_used =", tot_used)
    return subset_dict
            

if __name__ == '__main__':

    # Get path to cfg file used to make subsets - Yes... this ishokey,
    # but the cfg file specified is used by multiple modules. This way
    # I don't have to type, or remember, a long path to the file used
    # for each expt.I'll just have a build up of small cfg files in this dir
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg_path_file", type=str,
                    help="cfg_file_specifying_path")
    args = ap.parse_args()

    # Get pre-cfg file
    config_file = args.cfg_path_file
    config = configparser.ConfigParser()
    config.read(config_file)

    # Build path to cfg file
    cfg_root = config['PathStrs']['root']
    cfg_branch = config['PathStrs']['branch']
    cfg_leaf = config['PathStrs']['leaf']
    cfg_path = os.path.join(cfg_root,
                            cfg_branch,
                            cfg_leaf)
    

    # Find and Read cfg file
    print("Reading ", cfg_path)
    config = configparser.ConfigParser()
    config.read(cfg_path)

    # Get source dir for CalTech 101 data
    home = expanduser("~")
    data_root_dir = config['StorageDirectory']['data_root_dir']
    data_dir = config['StorageDirectory']['data_dir']
    subset_root_dir = config['StorageDirectory']['subset_root_dir']
    subset_dir = config['StorageDirectory']['subset_dir']
    
    subset_root_path = os.path.join(home, data_root_dir, data_dir,
                                    subset_root_dir)
    src_path = os.path.join(subset_root_path, subset_dir)
    print("Full Dataset Path:", src_path)

    print("Loading training set")
    train = pickle.load(open(os.path.join(src_path, "train"), 'rb'))
    train = convert(train)

    spc_list = [x.strip() for x in config['Subsets']['spc'].split(',')]
    suffix_list = [x.strip() for x in config['Subsets']['suffixes'].split(',')]
    
    for spc, suffix in itertools.product(spc_list, suffix_list):
        trgt_dir = "_".join([subset_dir, str(spc), suffix])
        trgt_path = os.path.join(subset_root_path, trgt_dir)
        print(trgt_path)

        # Note: shutil.copytree calls os.makedirs and will fail if trgt_path exists
        try:
            shutil.copytree(src_path, trgt_path, symlinks=False, ignore=None)
            print("Copying {} to {}".format(src_path, trgt_path))
        except FileExistsError:
            print("Skipping", trgt_path, ". File Already exists.")


        # Save training subset
        sd = get_subset(train, int(spc))
        pickle.dump(sd, open(os.path.join(trgt_path, 'train'), 'wb'))
        print("Saved to ", trgt_path)
    print("Done")
  
""" Sample cfg file:

[Subsets]
spc: 50
suffixes: a,b,c

[StorageDirectory]
data_root_dir: .keras/datasets
data_dir: cifar-100-python
subset_root_dir: cifar100_living_living
subset_dir: trgt_tasks

Notes:
data_root_dir is root dir where all raw data stored.
data_dir is dir where specific downloaded data is stored
subset_root_dir is dir where class-wise subset of downloaded data stored
subset_dir where sample-wise subset of subset data stored
"""
