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
from six.moves import cPickle

def make_sure_data_dir_exists(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        print ("Creation of the directory %s failed" % dir_path)
    else:
        print ("Successfully created the directory %s" % dir_path)

def make_data_dir(main_dir):

        done = False
        suffix = '_v0'

        curr_output_dir = main_dir # + '_' + self.gpu)
        while not done:
            if not os.path.isdir(curr_output_dir):
                make_sure_data_dir_exists(curr_output_dir)
                done = True
            else:
                # Make certain existing dataset not accidentally
                # over written
                version = int(suffix[2:]) + 1
                suffix = '_v' + str(version)
                curr_output_dir = main_dir + suffix
        print ("Saving results to %s" % curr_output_dir)
        return curr_output_dir


def get_subtasks(subtasks, data_path, dataset, samps_per_class):

    num_classes = len(subtasks)

    print("Loading data set")
    data_dict  = pickle.load(open(os.path.join(data_path, dataset), "rb"), encoding='latin1')
    classes_lists = pickle.load(open(os.path.join(data_path, 'meta'), 'rb'), encoding='latin1')
    coarse_classes = classes_lists['coarse_label_names']

    # Initialze info for subset_dict
    #  Images
    tot_images = samps_per_class*num_classes
    subset_data = np.zeros((tot_images, 3072))
    #  Image labels and meta-data
    subset_dict = dict()
    subset_dict['fine_labels'.encode('utf-8')] = []
    subset_dict['coarse_labels'.encode('utf-8')] = []
    subset_dict['filenames'.encode('utf-8')] = [] 
    subset_dict['batch_label'.encode('utf-8')] = "Subset training batch 1 of 1 - " + str(samps_per_class*num_classes)
    subset_dict['batch_label'.encode('utf-8')] += " samps per class"

    # Initialize dict to track number of samples used per class
    used_dict = defaultdict(int)

    tot_used = 0
    for filename, fine_num, coarse_num, data in zip(data_dict['filenames'],
                                                    data_dict['fine_labels'],
                                                    data_dict['coarse_labels'],
                                                    data_dict['data']):

        if coarse_classes[coarse_num] in subtasks:
            # Copy chosen sample
            subset_dict['fine_labels'.encode('utf-8')].append(fine_num)
            subset_dict['coarse_labels'.encode('utf-8')].append(coarse_num)
            subset_dict['filenames'.encode('utf-8')].append(filename)
            subset_data[tot_used,:] = data[:]
            
            # Update tracking variables
            tot_used += 1
            used_dict[coarse_num] += 1
        else:
            pass

    subset_dict['data'.encode('utf-8')] = subset_data
    for curr_class in sorted(used_dict):
        print (coarse_classes[curr_class],"(",curr_class,"): ", used_dict[curr_class])
    print("\n")

    return (subset_dict, classes_lists)

if __name__ == '__main__':

    # Get cfg file
    ap = argparse.ArgumentParser()
    ap.add_argument("infile", type = str,
                    help="file with input parameters")
    args = ap.parse_args()

    # Parse cfg file
    config = configparser.ConfigParser()
    config.read(args.infile)

    # Get classes for src and trgt tasks
    src_tasks = [x.strip() for x in config.get("Tasks","source").split(",")]
    trgt_tasks = [x.strip() for x in config.get("Tasks","target").split(",")]

    # Get data about datasets 
    class_type = config.get("MetaData","class_type")
    train_spc = config.getint("MetaData","samps_per_class_training")
    test_spc = config.getint("MetaData","samps_per_class_testing")

    # Get names for output dirs / expt names
    expt_name = config.get("output_names", "expt")
    source_name = config.get("output_names", "source")
    target_name = config.get("output_names", "target")
    
    # Get source dir for cifar 100 data
    home = expanduser("~")
    parent_dir = '.keras/datasets/'
    data_dir = 'cifar-100-python'
    data_path = os.path.join(home, parent_dir, data_dir)

    # Make ouput dirs 
    expt_path = os.path.join(home, parent_dir, data_dir, expt_name)
    #expt_path = make_data_dir(expt_path)
    try:
        os.makedirs(expt_path, exist_ok=True)
    except OSError:
        print ("Creation of the directory %s failed" % expt_path)
    src_path = os.path.join(expt_path, "src_tasks")
    src_path = make_data_dir(src_path)
    trgt_path = os.path.join(expt_path, "trgt_tasks")
    trgt_path = make_data_dir(trgt_path)

    for tasks, out_path in zip([src_tasks, trgt_tasks],[src_path, trgt_path]):
        for set_type, spc in zip(["train", "test"], [train_spc, test_spc]):
            (sd, meta) =  get_subtasks(tasks, data_path, set_type, spc)
            cPickle.dump(sd, open(os.path.join(out_path, set_type),'wb'))
            zz= cPickle.load(open(os.path.join(out_path, set_type),'rb'), encoding='bytes')
        pickle.dump(meta, open(os.path.join(out_path, "meta"),'wb'))


    '''
    python make_cifar_src_trgt_tasks.py ../dataset_info/cifar100_src_trgt_v1.cfg
    '''
