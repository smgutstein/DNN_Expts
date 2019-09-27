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

def make_data_dir(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError:
        print ("Creation of the directory %s failed" % dir_path)
    else:
        print ("Successfully created the directory %s" % dir_path)


def get_subtasks(subtasks, data_path, dataset, samps_per_class):

    num_classes = len(subtasks)

    print("Loading data set")
    data_dict  = pickle.load(open(os.path.join(data_path, dataset), "rb"), encoding='latin1')
    classes_lists = pickle.load(open(os.path.join(data_path, 'meta'), 'rb'), encoding='latin1')
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

    tot_used = 0
    for filename, fine_num, coarse_num, data in zip(data_dict['filenames'],
                                                    data_dict['fine_labels'],
                                                    data_dict['coarse_labels'],
                                                    data_dict['data']):

        if coarse_classes[coarse_num] in subtasks:
            # Copy chosen sample
            subset_dict['fine_labels'].append(fine_num)
            subset_dict['coarse_labels'].append(coarse_num)
            subset_dict['filenames'].append(filename)
            subset_data[tot_used,:] = data[:]
            
            # Update tracking variables
            tot_used += 1
            used_dict[coarse_num] += 1
        else:
            pass

    for curr_class in sorted(used_dict):
        print (coarse_classes[curr_class],"(",curr_class,"): ", used_dict[curr_class])
    print("\n\n")

    return (subset_dict, used_dict, classes_lists)

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
    
    class_type = config.get("MetaData","class_type")
    train_spc = config.getint("MetaData","samps_per_class_training")
    test_spc = config.getint("MetaData","samps_per_class_testing")

    expt_name = config.get("output_names", "expt")
    source_name = config.get("output_names", "source")
    target_name = config.get("output_names", "target")
    
    # Get source dir for cifar 100 data
    home = expanduser("~")
    parent_dir = '.keras/datasets/'
    data_dir = 'cifar-100-python'
    data_path = os.path.join(home, parent_dir, data_dir)

    # Make ouput dirs 
    expt_dir = expt_name
    expt_path = os.path.join(home, parent_dir, data_dir, expt_dir)
    src_path = os.path.join(expt_path, "src_tasks")
    trgt_path = os.path.join(expt_path, "trgt_tasks")
    make_data_dir(expt_path)
    make_data_dir(src_path)
    make_data_dir(trgt_path)


    for tasks, out_path in zip([src_tasks, trgt_tasks],[src_path, trgt_path]):
        for set_type, spc in zip(["train", "test"], [train_spc, test_spc]):
            (subset_dict, used_dict, meta) =  get_subtasks(tasks, data_path, set_type, spc)
            pickle.dump(subset_dict, open(os.path.join(out_path, set_type),'wb'))
        pickle.dump(meta, open(os.path.join(out_path, "meta"),'wb'))
            
    #(subset_dict, used_dict) =  get_subtasks(src_tasks, data_path, "train", train_spc)
    #(subset_dict, used_dict) =  get_subtasks(trgt_tasks, data_path, "train", train_spc)
    #(subset_dict, used_dict) =  get_subtasks(src_tasks, data_path, "test", test_spc)
    #(subset_dict, used_dict) =  get_subtasks(trgt_tasks, data_path, "test", test_spc)



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

    python make_cifar_src_trgt_tasks.py ./dataset_info/cifar100_src_trgt_v1.cfg
    '''
