from itertools import product
import os

# These global variables should really be turned into
# configuration file inputs, but I'm too lazy right now
expt_class = 'opt_tfer_expts'
expt_datasets = 'cifar_100_living_notliving_expts'
expt_arch = 'wide_resnet_28_10_arch'
spc_prefix = 'spc_'
src_epoch_prefix = 'src_epoch_'
src_prefix = 'src_net_'
tfer_tr_set_prefix = 'tr_set_'
res_dir = "results"
enc_dir = "cfg_dir/enc_cfg"
expt_dir = "cfg_dir/expt_cfg"
src_nets_dir = "src_nets/workshop_expts"

spc_list = [1,5,10,25,50,100,200]
src_epoch_list = [x for x in range(0,210,10)]
src_net_list = [0,1,2,3,4]
tfer_tr_set_list = ['a', 'b', 'c', 'd', 'e']


def make_cfg_file(spc, src_epoch, src_net, tr_set, machine_name, version):
    expt_file_str = "[ExptFiles]\n"
    expt_file_str += "data_loader: cifar100_trgt_living_vs_notliving_subset_"
    expt_file_str += str(spc) + '_' + str(tr_set)+ "\n"
    expt_file_str += "class_names: dataset_info/cifar100_dicts_all.pkl\n"
    expt_file_str += "encoding_cfg: " + os.path.join(enc_dir, "softmax_encoding_65.cfg") + "\n"
    expt_file_str += "root_expt_dir: " + os.path.join(res_dir, expt_class) + "\n"
    expt_file_str += "expt_dir: " + expt_datasets + "\n"
    expt_file_str += "expt_subdir: " + expt_arch + "/tfer_nets/" 
    subdir_suffix = os.path.join(spc_prefix + str(spc), src_epoch_prefix + str(src_epoch), 
                                 src_prefix + machine_name + "_" + version, "tr_set_" + str(tr_set))
    expt_file_str += subdir_suffix + "\n\n"
    
    expt_param_str = "[ExptParams]\n"
    expt_param_str += "epochs: 200\n"
    expt_param_str += "batch_size: 128\n"
    expt_param_str += "epochs_per_recording: 10\n\n"
    
    data_preproc_params_str = "[DataPreprocessParams]\n"
    data_preproc_params_str += "featurewise_center: True\n"
    data_preproc_params_str += "samplewise_center: False\n"
    data_preproc_params_str += "featurewise_std_normalization: True\n"
    data_preproc_params_str += "samplewise_std_normalization: False\n"
    data_preproc_params_str += "zca_whitening: False\n\n"
    
    data_aug_params_str = "[DataAugmentParams]\n"
    data_aug_params_str += "rotation_range: 0\n"
    data_aug_params_str += "width_shift_range: 4\n"
    data_aug_params_str += "height_shift_range: 4\n"
    data_aug_params_str += "brightness_range: None\n"
    data_aug_params_str += "shear_range: 0.0\n"
    data_aug_params_str += "zoom_range: 0.0\n"
    data_aug_params_str += "channel_shift_range: 0.0\n"
    data_aug_params_str += "fill_mode: nearest\n"
    data_aug_params_str += "cval: 0\n"
    data_aug_params_str += "horizontal_flip: True\n"
    data_aug_params_str += "vertical_flip: False\n\n"
    
    saved_params_str = "[SavedParams]\n"
    saved_params_str += "saved_set_dir: " + os.path.join(res_dir, expt_class, expt_datasets,
                                                         expt_arch, "src_nets") + "\n"
    saved_params_str += "saved_dir: " + os.path.join("_".join([machine_name, 
                                                               expt_datasets, 
                                                               version]),
                                                     "checkpoints") + "\n"
    saved_params_str += "saved_arch: init_arch.json\n"
    saved_params_str += "saved_iter: " + str(src_epoch)
    if str(src_epoch) != 'best':
        saved_params_str += "_Src_Epochs"
    saved_params_str += '\n'
    saved_params_str += "saved_encodings_iter: 1\n\n"

    trgt_task_params_str = "[TrgtTaskParams]\n"
    trgt_task_params_str += "num_reset_layers: 1\n"
    trgt_task_params_str += "penultimate_node_list: []\n"
    trgt_task_params_str += "output_activation: softmax\n"
    trgt_task_params_str += "class_names: dataset_info/cifar100_dicts_all.pkl\n"
    trgt_task_params_str += "encoding_cfg: " + os.path.join(enc_dir,
                                                            "softmax_encoding_35.cfg") +"\n"
    
    notes_str = "\n[Notes]\n"
    notes_str += "notes: Autogenerated file to tfer learn "
    notes_str += "non-living CIFAR100 from living CIFAR100\n\n"
    
    out_str = (expt_file_str + expt_param_str + data_preproc_params_str +
               data_aug_params_str + saved_params_str + trgt_task_params_str +
               notes_str)
    
    return out_str
               
def make_cfg_files_and_dirs():

    # Root dir for tfer cfg files
    root_dir = os.path.join('..', expt_dir,
                            expt_class,
                            expt_datasets,
                            expt_arch, 'tfer_nets' )

    # Root dir containing src nets
    src_net_root_dir =  os.path.join('..', res_dir,
                                     expt_class,
                                     expt_datasets,
                                     expt_arch,
                                     src_nets_dir)

    # Names of source net expts
    src_net_list = sorted(os.listdir(src_net_root_dir))

    ctr = 0
    ctr2 = 0
    # Loop over each (# of samples per class,
    #                 src net expts,
    #                 transfer tas training sets)
    for curr in product(spc_list, src_net_list, tfer_tr_set_list):
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
            new_leaf = os.path.join(spc_prefix + str(spc),
                                    src_epoch_prefix + str(src_epoch),
                                    src_prefix + str(src_name),
                                    tfer_tr_set_prefix + str(tr_set))

            cfg_dir = os.path.join(root_dir, new_leaf)    
            os.makedirs(cfg_dir, exist_ok = True)
            #print("Curr Dir: ", cfg_dir)

            # Create text of cfg file
            cfg_file_str = make_cfg_file(spc, src_epoch, src_net_dir, tr_set,
                                         machine_name, version)

            # Determine name of cfg file ... determine version/id number
            cfg_files = [x for x in os.listdir(cfg_dir) if 'tfer_net' in x]
            cfg_files = [int(x.split('tfer_net_')[1].split('.cfg')[0]) for x in cfg_files]
            cfg_files.sort()
            if len(cfg_files) == 0:
               cfg_version = 0
            else:
               cfg_version = max(cfg_files) + 1
            cfg_file_name = "tfer_net_" + str(cfg_version) + ".cfg"

            # Write cfg file
            #print("   Writing: ", os.path.join(cfg_dir, cfg_file_name))
            with open(os.path.join(cfg_dir, cfg_file_name), 'w') as f:
                f.write(cfg_file_str)
                ctr += 1
    print("\nTotal # of Files Created: ",ctr)
    print("Total # of Files Not Created: ",ctr2)

make_cfg_files_and_dirs()
