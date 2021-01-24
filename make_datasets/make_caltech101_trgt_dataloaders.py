import argparse
import configparser
import itertools
import os

from trgt_task_dataloader_strings_caltech101 import import_str, header_str
from trgt_task_dataloader_strings_caltech101 import doc_body_str, path_str, body_str


if __name__ == '__main__':
    # Get desired samples per class
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--cfg_root", type=str,
                    default="../cfg_dir/gen_cfg/opt_tfer_expts",
                    help="root dir for config files")
    ap.add_argument("-s", "--cfg_sub", type=str,
                    default="caltech101_living_notliving_expts",
                    help="dir for config files for set of expts")
    ap.add_argument("-l", "--cfg_leaf", type=str,
                    default="tfer_datasets/subsets.cfg",
                    help="dir for config files for set of expts")
    args = ap.parse_args()

    # Make target dirs and copy info from src dir
    config_file = os.path.join(args.cfg_root,
                               args.cfg_sub,
                               args.cfg_leaf)
    print("Reading ", config_file)
    config = configparser.ConfigParser()
    config.read(config_file)

    note = "Loads data for " + config['Notes']['note']
    doc_body_str = '    \"\"\"\n' + '    ' + note + doc_body_str + '    \"\"\"\n'

    data_root_dir = config['StorageDirectory']['data_root_dir']
    data_dir = config['StorageDirectory']['data_dir']
    subset_root_dir = config['StorageDirectory']['subset_root_dir']
    subset_dir = config['StorageDirectory']['subset_dir']

    # Get spc and training set id suffix for each data loader
    spc_list = [x.strip() for x in config['Subsets']['spc'].split(',')]
    suffix_list = [x.strip() for x in config['Subsets']['suffixes'].split(',')]

    for spc, suffix in itertools.product(spc_list, suffix_list):
        data_path = "_".join([path_str, str(spc), suffix + "'"])
        data_path += ")"

        out_str = import_str + header_str + doc_body_str + data_path + body_str
        out_path = '../dataset_loaders/'
        out_file = '_'.join([subset_root_dir,
                             subset_dir,
                             spc,
                             suffix]) + '.py'
        
        with open(os.path.join(out_path, out_file), 'w') as f:
            f.write(out_str)
            print("Wrote: ", os.path.join(out_path, out_file))
