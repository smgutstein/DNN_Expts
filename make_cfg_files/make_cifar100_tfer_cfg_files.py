from __future__ import print_function
import argparse
import configparser
from itertools import product
import os
from io import StringIO


def get_src_tfer_expt_root_dirs(config_v):

    path_dict = config_v['Prefixes']

    # Root dir for tfer cfg files
    root_dir = os.path.join('..', path_dict['expt_dir'],
                            path_dict['expt_class'],
                            path_dict['expt_datasets'],
                            path_dict['expt_arch'], 'tfer_nets')

    # Root dir containing src nets
    src_net_root_dir = os.path.join('..', path_dict['res_dir'],
                                    path_dict['expt_class'],
                                    path_dict['expt_datasets'],
                                    path_dict['expt_arch'],
                                    path_dict['src_nets_dir'])
    return root_dir, src_net_root_dir


def str2list(x):
    return x.strip().split('[')[1].split(']')[0].split(',')


def get_varying_params(config_v, src_net_root_dir):
    var_dict = config_v['Vars']

    # Lists of varying params
    spc_list = var_dict['spc']
    spc_list = str2list(spc_list)

    tr_set_list = var_dict['tr_set']
    tr_set_list = str2list(tr_set_list)

    # Names of source net expts
    src_net_list = sorted(os.listdir(src_net_root_dir))

    return spc_list, tr_set_list, src_net_list


def make_cfg_file_text(config_s, config_v,
                       spc, src_epoch,
                       src_net, tr_set,
                       machine_name, version):
    # Create deep copy of config obj using StringIO
    config_str = StringIO()
    config_s.write(config_str)

    config_str.seek(0)
    temp_config = configparser.ConfigParser()
    temp_config.read_file(config_str)

    prefix_dict = config_v["Prefixes"]
    expt_arch = prefix_dict["expt_arch"]
    net_group = prefix_dict['net_group']
    spc_dir = prefix_dict["spc_prefix"] + str(spc)
    src_epoch_dir = prefix_dict["src_epoch_prefix"] + str(src_epoch)

    dlg_prefix = prefix_dict["data_loader_group"]
    dlg = "_".join([dlg_prefix, str(spc), str(tr_set)])

    machine_dir = machine_name + "_" + version
    tr_set_dir = "tr_set_" + str(tr_set)

    temp_config["ExptFiles"]["data_loader"] = dlg
    temp_config["ExptFiles"]["expt_subdir"] = os.path.join(expt_arch,
                                                           net_group,
                                                           spc_dir,
                                                           src_epoch_dir,
                                                           machine_dir,
                                                           tr_set_dir)

    saved_set_dir = os.path.join(prefix_dict["res_dir"],
                                 prefix_dict["expt_class"],
                                 prefix_dict["expt_datasets"],
                                 expt_arch,
                                 prefix_dict["src_nets_dir"])
    temp_config["SavedParams"]["saved_set_dir"] = saved_set_dir

    checkpoint_dir = os.path.join("_".join([machine_name,
                                            version]),
                                  "checkpoints")
    temp_config["SavedParams"]["saved_dir"] = checkpoint_dir

    iter_str = str(src_epoch)
    if str(src_epoch) != 'best':
        iter_str += "_Src_Epochs"
    temp_config["SavedParams"]["saved_iter"] = iter_str

    return temp_config


def make_cfg_files(root_dir, var_lists):
    spc_list, tr_set_list, src_net_list = var_lists

    ctr = 0
    ctr2 = 0
    # Loop over each (# of samples per class,
    #                 src net expts,
    #                 transfer tas training sets)
    for curr in product(spc_list, src_net_list, tr_set_list):
        spc, src_net_dir, tr_set = curr

        # Get name of machine that ran src expt & a unique id number
        machine_name = src_net_dir.split('_')[0]
        version = src_net_dir.split('_')[-1]
        src_name = '_'.join([machine_name, version])

        # Get list of saved nets from given src nets
        saved_nets = [x[:-3]
                      for x in os.listdir(os.path.join(src_net_root_dir,
                                                       src_net_dir,
                                                       'checkpoints'))
                      if 'weights' in x and '.h5' in x]
        saved_nets.sort()
        best_epoch = [x.split('_')[-1] for x in saved_nets if 'best' in x][0]

        # Loop thru saved nets from given src expt
        for curr_src_net in saved_nets:

            # Identify particular saved epoch
            src_epoch = curr_src_net.split('_')[-1]
            if src_epoch == best_epoch and 'best' not in curr_src_net:
                ctr2 += 1
                print("Skipping ...", spc, src_epoch, src_name, tr_set)
                continue
            elif src_epoch == best_epoch:
                src_epoch = 'best'

            # Create dir path for cfg file
            new_leaf = os.path.join('spc_' + str(spc),
                                    'src_epoch_' + str(src_epoch),
                                    'src_net_' + str(src_name),
                                    'tr_set_' + str(tr_set))

            cfg_dir = os.path.join(root_dir, new_leaf)
            os.makedirs(cfg_dir, exist_ok=True)

            # Create text of cfg file
            text_cfg = make_cfg_file_text(config_s, config_v,
                                          spc, src_epoch,
                                          src_net_dir, tr_set,
                                          machine_name, version)

            # Determine name of cfg file ... determine version/id number
            cfg_files = [x for x in os.listdir(cfg_dir) if 'tfer_net' in x]
            cfg_files = [int(x.split('tfer_net_')[1].split('.cfg')[0])
                         for x in cfg_files]
            cfg_files.sort()
            if len(cfg_files) == 0:
                cfg_version = 0
            else:
                cfg_version = max(cfg_files) + 1
            cfg_file_name = "tfer_net_" + str(cfg_version) + ".cfg"

            # Write cfg file
            print("   Writing: ", os.path.join(cfg_dir, cfg_file_name))
            with open(os.path.join(cfg_dir, cfg_file_name), 'w') as f:
                text_cfg.write(f)
                ctr += 1

    print("\nTotal # of Files Created: ", ctr)
    print("Total # of Files Not Created: ", ctr2)


if __name__ == '__main__':
    # Get desired samples per class
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--cfg_root", type=str,
                    default="../cfg_dir/gen_cfg/opt_tfer_expts",
                    help="root dir for config files")
    ap.add_argument("-b", "--cfg_branch", type=str,
                    default="cifar_100_living_notliving_expts",
                    help="dir for config files for set of expts")
    ap.add_argument("-l", "--cfg_leaf", type=str,
                    default="tfer_datasets",
                    help="dir for config files for set of expts")
    ap.add_argument("-s", "--cfg_subset_file", type=str,
                    default="subsets.cfg",
                    help="dir for config files for set of expts")

    ap.add_argument("-c", "--cfg_skel", type=str,
                    default="tfer_net_skeleton_0.cfg",
                    help="skeleton for expt cfg files")
    ap.add_argument("-v", "--cfg_var", type=str,
                    default="tfer_net_vars_0.cfg",
                    help="dir for config files for set of expts")
    args = ap.parse_args()

    # Find and Read cfg file with info on trgt task datasets/expts 
    config_dir = os.path.join(args.cfg_root,
                              args.cfg_branch,
                              args.cfg_leaf)
    config_file = os.path.join(config_dir, args.cfg_subset_file)
    print("Reading ", config_file)
    config = configparser.ConfigParser()
    config.read(config_file)

    # Find and Read cfg file with info common to all expts
    config_skel_file = os.path.join(config_dir, args.cfg_skel)
    print("Reading ", config_skel_file)
    config_s = configparser.ConfigParser()
    config_s.read(config_skel_file)

    # Find and Read cfg file with info that varies by expt
    config_var_file = os.path.join(config_dir, args.cfg_var)
    print("Reading ", config_var_file)
    config_v = configparser.ConfigParser()
    config_v.read(config_var_file)

    root_dir, src_net_root_dir = get_src_tfer_expt_root_dirs(config_v)
    var_lists = get_varying_params(config_v, src_net_root_dir)
    make_cfg_files(root_dir, var_lists)

    '''
    Relies on 3 cfg files: 
    1. A cfg giving info on data used for trgt task, particularly
    samples per class for trgt training sets and number of diffferent 
    trgt training sets for each spc z.b.:
    
    [Subsets]
    spc: 1,5,10,25,50,100,200,250
    suffixes: a,b,c,d,e

    [StorageDirectory]
    data_root_dir: .keras/datasets
    data_dir: cifar-100-python
    subset_root_dir: cifar100_living_notliving
    subset_dir: trgt_tasks

    [Notes]
    note: CIFAR 100 living vs not living datasets

    2. A cfg giving skeleton of cfg file for ALL expts z.b.:
    
    [ExptFiles]
    class_names: dataset_info/cifar100_dicts_all.pkl
    encoding_cfg: cfg_dir/enc_cfg/softmax_encoding_65.cfg
    root_expt_dir: results/opt_tfer_expts
    expt_dir: cifar_100_living_notliving_expts
    expt_subdir: wide_resnet_28_10_arch/tfer_nets
    
    [NetParams]
    arch_module: cifar100_wide_resnet
    output_activation: softmax
    optimizer_cfg: cfg_dir/opt_cfg/optimizer_sgd_wrn.cfg
    loss_fnc: categorical_crossentropy
    N: 4
    k: 10
    dropout: 0.00
    
    [ExptParams]
    epochs: 200
    batch_size: 128
    epochs_per_recording: 10
    
    [DataPreprocessParams]
    featurewise_center: True
    samplewise_center: False
    featurewise_std_normalization: True
    samplewise_std_normalization: False
    zca_whitening: False
    
    [DataAugmentParams]
    rotation_range: 0
    width_shift_range: 4
    height_shift_range: 4
    brightness_range: None
    shear_range: 0.0
    zoom_range: 0.0
    channel_shift_range: 0.0
    fill_mode: nearest
    cval: 0
    horizontal_flip: True
    vertical_flip: False
    
    
    [SavedParams]
    saved_arch: init_arch.json
    saved_encodings_iter: 1
    
    [TrgtTaskParams]
    num_reset_layers: 1
    penultimate_node_list: []
    output_activation: softmax
    class_names: dataset_info/cifar100_dicts_all.pkl
    encoding_cfg: cfg_dir/enc_cfg/softmax_encoding_35.cfg
    
    [Notes]
    notes: Autogenerated file to degenerate tfer learn living CIFAR100 from living CIFAR100

    3. A cfg file with attributes that vary across each expt and some const prefixes &
    suffixes used to create various directory names z.b.:
       
    [Vars]
    spc: [1,5,10,25,50,100,200,250]
    tr_set: [a,b,c,d,e]
    
    [Prefixes]
    expt_class: opt_tfer_expts
    expt_datasets: cifar_100_living_notliving_expts
    expt_datasets_b: cifar_100_living_notliving_expts
    expt_arch: wide_resnet_28_10_arch
    spc_prefix: spc_
    src_epoch_prefix: src_epoch_
    src_prefix: src_net_
    tfer_tr_set_prefix: tr_set_
    res_dir: results
    enc_dir: cfg_dir/enc_cfg
    expt_dir: cfg_dir/expt_cfg
    opt_dir: cfg_dir/opt_cfg
    src_nets_dir: src_nets/workshop_expts
    data_loader_group: cifar100_trgt_living_vs_notliving_subset
    net_group: tfer_nets

    '''
