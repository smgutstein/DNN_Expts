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

from generate_cifar100_trgt_subset_dataloaders import head_str, tail_str


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
                    default = "cifar100_living_not_living",
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
    ds_suffix_list = [x.strip() for x in config['Subsets']['suffixes'].split(',')]
    
    for spc,ds_suffix in itertools.product(spc_list, ds_suffix_list):
        suffix = str(spc) + '_' + str(ds_suffix)
        outstr = head_str + " + \'" + suffix +"\'" + tail_str
        outpath  = '../dataset_loaders/'
        outfile = 'cifar100_trgt_living_vs_notliving_subset_' + suffix + '.py'
        with open(os.path.join(outpath,outfile), 'w') as f:
            f.write(outstr)
            print ("Wrote ", os.path.join(outpath,outfile))
    
