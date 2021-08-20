import argparse
import configparser
import itertools
import os

from trgt_task_dataloader_strings import import_str1, import_str2, header_str
from trgt_task_dataloader_strings import doc_body_str, path_str, body_str


if __name__ == '__main__':
    # NOTE: This uses same cfg files as make_spc_training_sets.py  
    # Get desired samples per class
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
    load_module = config['PathStrs']['load_module']
    
    # Find and Read cfg file    
    print("Reading ", cfg_path)
    config = configparser.ConfigParser()
    config.read(cfg_path)

    # Make dataloader file
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
        data_path = path_str
        data_path += ")\n\n"

        out_str = import_str1 + load_module + import_str2 + \
                  header_str + doc_body_str + data_path + body_str
        out_path = '../dataset_loaders/'
        out_file = '_'.join([subset_root_dir,
                             subset_dir,
                             spc,
                             suffix]) + '.py'
        
        with open(os.path.join(out_path, out_file), 'w') as f:
            f.write(out_str)
            print("Wrote: ", os.path.join(out_path, out_file))
