import argparse
import configparser
import itertools
import os

head_str = '''
from __future__ import absolute_import
import os
import sys
file_dir = os.path.dirname(os.path.realpath(__file__))
if file_dir not in sys.path:
    sys.path.append(file_dir)
from cifar import load_batch
from keras.utils.data_utils import get_file
from keras import backend as K
import numpy as np
from os.path import expanduser
home = expanduser("~")

def load_data(label_mode='fine'):
    \"\"\"Loads trgt tasks for CIFAR100 living_vs_notliving datasets.

    # Arguments
        label_mode: one of \"fine\", \"coarse\".

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    # Raises
        ValueError: in case of invalid `label_mode`.
    \"\"\"
    if label_mode not in [\'fine\', \'coarse\']:
        raise ValueError(\'`label_mode` must be one of `\"fine\"`, `\"coarse\"`.')

    path = os.path.join(home,\'.keras/datasets/\', \'cifar-100-python\',
                        \'Living_vs_Not_Living\', \'trgt_tasks_\' '''

tail_str = ''' ) 

    fpath = os.path.join(path, \'train\')
    x_train, y_train = load_batch(fpath, label_key=label_mode + \'_labels\')

    fpath = os.path.join(path, 'test')
    x_test, y_test = load_batch(fpath, label_key=label_mode + \'_labels\')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # Rescale raw data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.
    x_test /= 255.

    if K.image_data_format() == \'channels_last\':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)
'''


def write_cfg_files(config_infile, config_outfile):
    # Write cfg files used for all expts in batch
    # Initialize/Get file structure data
    src_net_root_dir =  config_infile['RootDirs']['src_net_root_dir']
    trgt_cfg_root_dir = config_infile['RootDirs']['trgt_cfg_root_dir']

    expt_subdir_base = config_infile['BaseDirStrs']['expt_subdir_base']
    saved_dir_base = config_infile['BaseDirStrs']['saved_dir_base']
    saved_iter_base = config_infile['BaseDirStrs']['saved_iter_base']
    data_loader_base = config_infile['BaseDirStrs']['data_loader_base']
    
    file_name_prefix = config_infile['CfgFileSubstrings']['file_name_prefix']
    file_name_midfix = config_infile['CfgFileSubstrings']['file_name_midfix']
    file_name_suffix = config_infile['CfgFileSubstrings']['file_name_suffix']

    # Create lists of varying params
    src_net_list = config_infile['ExptParams']['src_net_list'].split(',')
    spc_list = config_infile['ExptParams']['spc_list'].split(',')
    src_epoch_list = config_infile['ExptParams']['src_epoch_list'].split(',')
    trgt_train_id_list = config_infile['ExptParams']['trgt_train_id_list'].split(',')

    src_net_list = [x.strip('[] ') for x in src_net_list] 
    spc_list = [x.strip('[] ') for x in spc_list]
    src_epoch_list = [x.strip('[] ') for x in src_epoch_list]
    trgt_train_id_list = [x.strip('[] ') for x in trgt_train_id_list]
    param_lists = [src_net_list, spc_list, src_epoch_list, trgt_train_id_list]

    # Write all cfg files
    for curr_SRC_NET, curr_SPC, curr_SRC_EPOCH, curr_TR_ID in list(itertools.product(*param_lists)):
        file_name = file_name_prefix + file_name_midfix + curr_SRC_EPOCH + '_' + curr_TR_ID + file_name_suffix
        cfg_dir_path = os.path.join(trgt_cfg_root_dir, 
                                    'src_net_' + curr_SRC_NET, curr_SPC + 'spc')    
        os.makedirs(cfg_dir_path, exist_ok = True)

        cfg_file_path = os.path.join(cfg_dir_path, file_name)

        config_outfile['ExptFiles']["expt_subdir"] = os.path.join(expt_subdir_base, 
                                                          'src_net_' + curr_SRC_NET,
                                                          curr_SPC + 'spc')
        config_outfile['SavedParams']['saved_dir'] = saved_dir_base + curr_SRC_NET + '/checkpoints'
        config_outfile['SavedParams']['saved_iter'] = curr_SRC_EPOCH + saved_iter_base
        config_outfile['TrgtTaskParams']['data_loader'] = data_loader_base + curr_SPC + '_' + curr_TR_ID

        config_outfile.write(open(cfg_file_path,'w'))
        print(cfg_file_path)


def write_shell_scripts(config_infile):
    # Write nested shell scripts used to run full batch of expts

    # Get data for shell script dir
    exec_sh_root_dir = config_infile['ExecParams']['exec_sh_root_dir']
    exec_sh_spc_dir = config_infile['ExecParams']['exec_sh_spc_dir']
    exec_sh_file_prefix = config_infile['ExecParams']['exec_sh_file_prefix']
    exec_sh_file_suffix = config_infile['ExecParams']['exec_sh_file_suffix']
    
    # If necessary make shell script dir
    os.makedirs(exec_sh_root_dir, exist_ok=True)

    # Create cmd strings to be exectued by shell scripts
    sh_str = "#!  /bin/bash \n\n"
    cmd_str_root = "python run_expt.py "
    cmd_str_suffix = " > /dev/null 2>&1"

    # Create arguments for cmd strings (i.e. arguments for run_expt.py
    arg_root_dir = os.path.join(config_infile['RootDirs']['trgt_cfg_root_dir'], "src_net_")
    arg_file_pre_str = config_infile['CfgFileSubstrings']['file_name_prefix'] + \
                       config_infile['CfgFileSubstrings']['file_name_midfix']
    arg_file_end_str = '.cfg --nocheckpoint'

    # Create root shell script file
    full_batch_name = 'tfer_net_batch.sh'
    full_batch_path = os.path.join(exec_sh_root_dir, full_batch_name)
    full_batch_file = open(full_batch_path, 'w')
    full_batch_file.write(sh_str)
    echo_str = "echo 'time " + full_batch_path +"' `date` \n"
    full_batch_file.write(echo_str)

    # Get list of params over which expts vary
    src_net_list = config_infile['ExptParams']['src_net_list'].split(',')
    spc_list = config_infile['ExptParams']['spc_list'].split(',')
    src_epoch_list = config_infile['ExptParams']['src_epoch_list'].split(',')
    trgt_train_id_list = config_infile['ExptParams']['trgt_train_id_list'].split(',')

    src_net_list = [x.strip('[] ') for x in src_net_list] 
    spc_list = [x.strip('[] ') for x in spc_list]
    src_epoch_list = [x.strip('[] ') for x in src_epoch_list]
    trgt_train_id_list = [x.strip('[] ') for x in trgt_train_id_list]

    # Build shell scripts
    for curr_SRC_NET in src_net_list:
        # Create shell scripts for given source net
        curr_src_dir = os.path.join(exec_sh_root_dir, "src_net_" + curr_SRC_NET)
        os.makedirs(curr_src_dir, exist_ok=True)

        src_batch_name = 'batch_src_' + curr_SRC_NET + '.sh'
        src_batch_path = os.path.join(curr_src_dir, src_batch_name)
        print (src_batch_path)
        src_batch_file = open(src_batch_path, 'w')
        src_batch_file.write(sh_str)
        echo_str = "echo '  " + src_batch_path +"' `date` \n"
        src_batch_file.write(echo_str)

        for curr_SPC in spc_list:
            # Create shell scripts for give (source net, samples per class)
            curr_spc_dir = os.path.join(curr_src_dir, curr_SPC + 'spc')
            os.makedirs(curr_spc_dir, exist_ok=True)

            spc_batch_name = 'batch_' + curr_SPC + 'spc' + '.sh'
            spc_batch_path = os.path.join(curr_spc_dir, spc_batch_name)
            print ("   ", spc_batch_path)
            spc_batch_file = open(spc_batch_path, 'w')
            spc_batch_file.write(sh_str)
            echo_str = "echo '    " + spc_batch_path +"' `date` \n"
            spc_batch_file.write(echo_str)

            for curr_SRC_EPOCH in src_epoch_list:
                # Create shell scripts for give (source net, samples per class, source task training epochs)
                curr_src_epoch_dir = os.path.join(curr_spc_dir, 'src_epoch_' + curr_SRC_EPOCH)
                os.makedirs(curr_src_epoch_dir, exist_ok=True)

                src_epoch_batch_name = 'batch_src_epoch_' + curr_SRC_EPOCH + '.sh'
                src_epoch_batch_path = os.path.join(curr_src_epoch_dir, 
                                                    src_epoch_batch_name)
                print ("      ", src_epoch_batch_path)
                src_epoch_batch_file = open(src_epoch_batch_path, 'w')
                src_epoch_batch_file.write(sh_str)
                echo_str = "echo '      " + src_epoch_batch_path +"' `date` \n"
                src_epoch_batch_file.write(echo_str)

                arg_str_prefix = os.path.join(arg_root_dir + curr_SRC_NET, 
                                              curr_SPC + 'spc')

                for curr_TR_ID  in trgt_train_id_list:
                    # Create shell scripts for give (source net, samples per class,
                    #                                source task training epochs,
                    #                                target task training set)
                    # NOTE: This innermost loop generates cmds for leaf shell script
                    echo_str = "echo '          src_net" + curr_SRC_NET 
                    echo_str += "_" + curr_SPC + 'spc_epoch_' + curr_SRC_EPOCH + curr_TR_ID +"'  `date` \n"
                    src_epoch_batch_file.write(echo_str)

                    cfg_file_name = arg_file_pre_str + curr_SRC_EPOCH + "_" + curr_TR_ID + arg_file_end_str
                    arg_str = os.path.join(arg_str_prefix, cfg_file_name)
                    cmd_str = cmd_str_root + arg_str + cmd_str_suffix + '\n'
                    src_epoch_batch_file.write(cmd_str)

                src_epoch_batch_file.close()
                os.chmod(src_epoch_batch_path, 0o755)
                spc_batch_file.write(src_epoch_batch_path + '\n')

            spc_batch_file.close()
            os.chmod(spc_batch_path, 0o755)
            src_batch_file.write(spc_batch_path + '\n')

        src_batch_file.close()
        os.chmod(src_batch_path, 0o755)
        full_batch_file.write(src_batch_path + '\n')

    full_batch_file.close()
    os.chmod(full_batch_path, 0o755)

def write_dataloaders(config_infile):
    # Crate dataloaders by Calling module that creates dataloader

    # Create lists of varying params for different datasets
    spc_list = config_infile['ExptParams']['spc_list'].split(',')
    trgt_train_id_list = config_infile['ExptParams']['trgt_train_id_list'].split(',')

    #cmd_str = 'python ./make_datasets/generate_cifar100_trgt_subset_dataloaders.py'
    outpath  = './dataset_loaders/'
    for trgt_dataset in list(itertools.product(spc_list, trgt_train_id_list)):
        #print(trgt_dataset)
        curr_spc = trgt_dataset[0].strip()
        curr_dataset = trgt_dataset[1].strip()
        suffix = "_".join([curr_spc, curr_dataset])
        #print (" ".join([cmd_str, arg_str]))
        outstr = head_str + " + \'" + suffix +"\'" + tail_str
        outfile = 'cifar100_trgt_living_vs_notliving_subset_' + suffix + '.py'
        with open(os.path.join(outpath,outfile), 'w') as f:
            f.write(outstr)
            print("Wrote ",os.path.join(outpath,outfile))
    

if __name__ == "__main__":
    config_root_dir = "./cfg_dir/gen_cfg/"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("SkeletonCfg", help="skeleton cfg file for series of expts")
    parser.add_argument("NonSkeletonCfg",
               help="information for constructing cfg files for expts, not in skeleton")
    parser.add_argument("--Major", type=str, default = "opt_tfer_expts",
               help="Directory for all expts in series")
    parser.add_argument("--Data", type=str, default = "cifar_100_living_notliving_expts",
               help="Directory for all expts using given datasets")
    parser.add_argument("--Arch", type=str, default = "prelim_arch",
               help="Directory for all expts using given net architecture")
    parser.add_argument("--Src", action="store_true",
               help="Directory for all expts using given net architecture")
    
    args = parser.parse_args()
    non_skeleton_cfg = args.NonSkeletonCfg
    skeleton_cfg = args.SkeletonCfg
    major_expts = args.Major
    dataset = args.Data
    arch = args.Arch
    src_nets = args.Src
    if src_nets:
       config_leaf_dir = "src_nets"
    else:
       config_leaf_dir = "tfer_nets"

    skeleton_cfg_file = os.path.join(config_root_dir, major_expts,
                                 dataset, arch, config_leaf_dir, skeleton_cfg)
    non_skeleton_cfg_file = os.path.join(config_root_dir, major_expts,
                                 dataset, arch, config_leaf_dir, non_skeleton_cfg)

    skel_config = configparser.ConfigParser()
    non_skel_config = configparser.ConfigParser()
    
    skel_config.read(skeleton_cfg_file)
    non_skel_config.read(non_skeleton_cfg_file)

    write_cfg_files(non_skel_config, skel_config)
    write_shell_scripts(non_skel_config)
    write_dataloaders(non_skel_config)
    
    #python make_cfg_and_sh_files.py base.cfg prelim_series.cfg
