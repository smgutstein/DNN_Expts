from __future__ import print_function
import argparse
from collections import defaultdict
import configparser
import errno
import itertools
import numpy as np
import pickle
import os
from os.path import expanduser
import shutil



def convert(data):
    # Converts strings of form b'my_string' to 'my_string'
    # (i.e. converts bytes to str)
    if isinstance(data, bytes):  return data.decode()
    if isinstance(data, dict):   return dict(map(convert, data.items()))
    if isinstance(data, tuple):  return tuple(map(convert, data))
    if isinstance(data, list):   return list(map(convert, data))
    return data

def get_subset(samps_per_class):

    print("Loading training set")
    train  = pickle.load(open(os.path.join(src_path,"train"),'rb'))
    train = convert(train)
    num_classes = len(set(train['fine_labels']))
    
    # Initialze info for subset_dict
    subset_data = np.zeros((samps_per_class*num_classes, 3072)) # 32*32*3=3072
    subset_dict = dict()
    subset_dict['fine_labels'] = []
    subset_dict['coarse_labels'] = []
    subset_dict['filenames'] = [] 
    subset_dict['batch_label'] = "Subset training batch 1 of 1 - " + str(samps_per_class*num_classes)
    subset_dict['batch_label'] += " samps per class"
    
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
            subset_data[tot_used,:] = train['data'][curr_candidate,:]
            
            # Update tracking variables
            tot_used += 1
            used_dict[curr_candidate_class] += 1
        else:
            pass
        
        # Proceed to next candidate element
        curr_candidate = candidate_list.pop()
        
    subset_dict['data'] = subset_data
    print ("tot_used =", tot_used)
    return subset_dict
            

if __name__ == '__main__':

    config_root_dir = "../cfg_dir/gen_cfg/opt_tfer_expts"
    config_leaf = "tfer_datasets/subsets.cfg"

    data_root_dir = '.keras/datasets'

    # Get desired samples per class
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_subdir", type = str,
                    default = "cifar_100_living_notliving_expts",
                    help="config file sub-directory")
    ap.add_argument("--data_dir", type = str, 
                    default='cifar-100-python',
                    help="source directory")
    ap.add_argument("--subset_root_dir", type=str,
                    default = "Living_vs_Not_Living",
                    help="sub dir for all subsets")
    ap.add_argument("--subset_dir", type=str,
                    default = "trgt_tasks",
                    help="sub dir for trgt subset")
    args = ap.parse_args()

    # Get source dir for cifar 100 data
    home = expanduser("~")
    subset_root_path = os.path.join(home, data_root_dir, args.data_dir,
                                    args.subset_root_dir)
    src_path = os.path.join(subset_root_path, args.subset_dir)

    # Make target dirs and copy info from src dir
    cfg_subdir = args.cfg_subdir
    config_file = os.path.join(config_root_dir, 
                              cfg_subdir, 
                              config_leaf)
    config = configparser.ConfigParser()
    config.read(config_file)
    spc_list = [x.strip() for x in config['Subsets']['spc'].split(',')]
    suffix_list = [x.strip() for x in config['Subsets']['suffixes'].split(',')]
    for spc,suffix in itertools.product(spc_list, suffix_list):

        trgt_dir = "_".join([args.subset_dir, str(spc), suffix])
        trgt_path = os.path.join(subset_root_path, trgt_dir)
        print(trgt_path)

        # Note: shutil.copytree calls os.makedirs and will fail if trgt_path exists
        shutil.copytree(src_path, trgt_path, symlinks=False, ignore=None)

        # Save training subset
        sd = get_subset(int(spc))
        pickle.dump(sd, open(os.path.join(trgt_path, 'train'),'wb'))
        print ("Saved to ", trgt_path)
    print("Done")
  