import argparse
from collections import defaultdict
import configparser
import numpy as np
import pickle
import os
from os.path import expanduser
from six.moves import cPickle
import sys


def make_sure_data_dir_exists(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        print("Creation of the directory %s failed" % dir_path)
    else:
        print("Successfully created the directory %s" % dir_path)


def make_data_dir(main_dir):
    done = False
    suffix = '_v0'

    curr_output_dir = main_dir  # + '_' + self.gpu)
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
    print("Saving results to %s" % curr_output_dir)
    return curr_output_dir


def get_subtasks(subtasks, data_path, dataset, samps_per_class):
    num_classes = len(subtasks)

    print("\nLoading", os.path.join(data_path, dataset))
    data_dict = pickle.load(open(os.path.join(data_path, dataset), "rb"), encoding='latin1')
    classes_lists = pickle.load(open(os.path.join(data_path, 'meta'), 'rb'), encoding='latin1')
    coarse_classes = classes_lists['coarse_label_names']

    # Check validity of subtasks list
    coarse_set = set(coarse_classes)
    subt_set = set(subtasks)
    found_set = coarse_set.intersection(subt_set)
    if len(found_set) != len(subt_set):
       missing_set = subt_set - found_set
       missing_list = sorted(list(missing_set))
       print ("Error: Could not find the following classes:")
       for x in missing_list:
           print("    ",x)
       print(" ")
       sys.exit()
      
    # Initialze info for subset_dict
    #  Images
    tot_images = samps_per_class * num_classes
    subset_data = np.zeros((tot_images, 3072))
    #  Image labels and meta-data
    subset_dict = dict()
    subset_dict['fine_labels'.encode('utf-8')] = []
    subset_dict['coarse_labels'.encode('utf-8')] = []
    subset_dict['filenames'.encode('utf-8')] = []
    subset_dict['batch_label'.encode('utf-8')] = "Subset training batch 1 of 1 - " + str(samps_per_class * num_classes)
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
            subset_data[tot_used, :] = data[:]

            # Update tracking variables
            tot_used += 1
            used_dict[coarse_num] += 1
        else:
            pass

    subset_dict['data'.encode('utf-8')] = subset_data
    print("Total Classes Used: ", len(used_dict), "out of", len(found_set))
    for curr_class in sorted(used_dict):
        print("  ",coarse_classes[curr_class], "(", curr_class, "): ", used_dict[curr_class])

    return subset_dict, classes_lists


if __name__ == '__main__':

    # Get cfg file
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--infile", type=str,
                    default="../dataset_info/caltech101_src_trgt_v1.cfg",
                    help="file with input parameters")
    args = ap.parse_args()

    # Parse cfg file
    config = configparser.ConfigParser()
    config.read(args.infile)

    # Get classes for src and trgt tasks
    src_tasks = [x.strip() for x in config.get("Tasks", "source").split(",")]
    trgt_tasks = [x.strip() for x in config.get("Tasks", "target").split(",")]

    # Get data about datasets 
    class_type = config.get("MetaData", "class_type")
    train_spc = config.getint("MetaData", "samps_per_class_training")
    test_spc = config.getint("MetaData", "samps_per_class_testing")

    # Get names for output dirs / expt names
    expt_name = config.get("output_names", "expt")
    source_name = config.get("output_names", "source")
    target_name = config.get("output_names", "target")

    # Get source dir for caltech 101 data
    home = expanduser("~")
    parent_dir = '.keras/datasets/'
    data_dir = 'caltech-101'
    data_path = os.path.join(home, parent_dir, data_dir)

    # Make ouput dirs 
    expt_path = os.path.join(data_path, expt_name)
    try:
        os.makedirs(expt_path, exist_ok=True)
    except OSError:
        print("Creation of the directory %s failed" % expt_path)

    src_path = os.path.join(expt_path, "src_tasks")
    src_path = make_data_dir(src_path)
    trgt_path = os.path.join(expt_path, "trgt_tasks")
    trgt_path = make_data_dir(trgt_path)

    for tasks, out_path in zip([src_tasks, trgt_tasks], [src_path, trgt_path]):
        for set_type, spc in zip(["train", "test"], [train_spc, test_spc]):
            (sd, meta) = get_subtasks(tasks, data_path, set_type, spc)
            cPickle.dump(sd, open(os.path.join(out_path, set_type), 'wb'))
            print("Writing ", os.path.join(out_path, set_type))
        # Metadata does not vary between training & testing sets
        pickle.dump(meta, open(os.path.join(out_path, "meta"), 'wb'))

    '''
    python make_cifar_src_trgt_tasks.py -i ../dataset_info/cifar100_src_trgt_v1.cfg
    
    Sample cfg:
    
    [MetaData]
    class_type: coarse
    samps_per_class_training: 2500
    samps_per_class_testing: 500

    [Tasks]
    source: aquatic_mammals, fish, flowers, fruit_and_vegetables, insects, large_carnivores,
            large_omnivores_and_herbivores,
            medium_mammals, non-insect_invertebrates, people, reptiles,
            small_mammals, trees

    target: food_containers, household_electrical_devices, household_furniture,
            large_man-made_outdoor_things, large_natural_outdoor_scenes, vehicles_1,
            vehicles_2

    [output_names]
    expt: cifar100_living_not_living
    source: Animals_Plants
    target: Non_Living_Things

    '''
