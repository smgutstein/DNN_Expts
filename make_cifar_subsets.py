from __future__ import print_function
import argparse
from collections import defaultdict
import errno
import numpy as np
import pickle
import os
from os.path import expanduser
import shutil

num_classes = 100


def get_subset(samps_per_class):

    print("Loading training set")
    train  = pickle.load(open(os.path.join(src_path,"train"),'r'))
    
    # Initialze info for subset_dict
    subset_data = np.zeros((samps_per_class*num_classes, 3072))
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
    curr_candidate = 0
    
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
        curr_candidate += 1
        
    subset_dict['data'] = subset_data
    print ("tot_used =", tot_used)
    return subset_dict
            

if __name__ == '__main__':

    # Get desired samples per class
    ap = argparse.ArgumentParser()
    ap.add_argument("spc", type = int,
                    help="samples per class")
    args = ap.parse_args()

    # Get source dir for cifar 100 data
    home = expanduser("~")
    dataset_dir = '.keras/datasets/'
    src_dir = 'cifar-100-python'
    src_path = os.path.join(home, dataset_dir, src_dir)

    # Make target dir and copy info from src dir
    trgt_dir = 'cifar-100-python' + "_" + str(args.spc)
    trgt_path = os.path.join(home, dataset_dir, trgt_dir)

    # Note: shutil.copytree calls os.makedirs and will fail if trgt_path exists
    shutil.copytree(src_path, trgt_path, symlinks=False, ignore=None)

    # Save training subset
    sd = get_subset(args.spc)
    pickle.dump(sd, open(os.path.join(trgt_path, 'train'),'wb'))
    print("Done")
