import argparse
from collections import defaultdict
import configparser
import errno
import json
import numpy as np
import pickle
import os
from os.path import expanduser
import shutil



def get_subtasks(subtasks, dataset, samps_per_class, num_classes):

    print("Loading training set")
    data_dict  = pickle.load(open(dataset,"rb"), encoding='latin1')
    classes_lists = pickle.load(open('meta','rb'), encoding='latin1')
    coarse_classes = classes_lists['coarse_label_names']

    # Initialze info for subset_dict
    tot_images = samps_per_class*num_classes
    subset_data = np.zeros((tot_images, 3072))
    subset_dict = dict()
    subset_dict['fine_labels'] = []
    subset_dict['coarse_labels'] = []
    subset_dict['filenames'] = [] 
    subset_dict['batch_label'] = "Subset training batch 1 of 1 - " + str(samps_per_class*num_classes)
    subset_dict['batch_label'] += " samps per class"

    # Initialize dict to track number of samples used per class
    used_dict = defaultdict(int)

    curr_candidate = 0
    for filename, batch, fine_nums, coarse_nums, data in zip(data_dict['filenames'],
                                                             data_dict['batch_label'],
                                                             data_dict['fine_labels'],
                                                             data_dict['coarse_labels'],
                                                             data_dict['data']):

        if coarse_classes[coarse_num] in subtasks:    
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



if __name__ == '__main__':

    # Get desired samples per class
    ap = argparse.ArgumentParser()
    ap.add_argument("infile", type = str,
                    help="file with input parameters")
    args = ap.parse_args()

    config = configparser.ConfigParser()
    config.read(args.infile)
    src_tasks = [x.strip() for x in config.get("Tasks","source").split(",")]
    trgt_tasks = [x.strip() for x in config.get("Tasks","target").split(",")]

    
    # Get source dir for cifar 100 data
    home = expanduser("~")
    dataset_dir = '.keras/datasets/'
    src_dir = 'cifar-100-python'
    src_path = os.path.join(home, dataset_dir, src_dir)

    print("Loading training set")
    train  = pickle.load(open(os.path.join(src_path,"train"),'r'))
    
    print("Loading testing set")
    test  = pickle.load(open(os.path.join(src_path,"test"),'r'))
    

    '''
    # Make target dir and copy info from src dir
    trgt_dir = 'cifar-100-python' + "_" + str(args.spc)
    trgt_path = os.path.join(home, dataset_dir, trgt_dir)

    # Note: shutil.copytree calls os.makedirs and will fail if trgt_path exists
    shutil.copytree(src_path, trgt_path, symlinks=False, ignore=None)

    # Save training subset
    sd = get_subset(args.spc)
    pickle.dump(sd, open(os.path.join(trgt_path, 'train'),'wb'))
    print("Done")
    '''
