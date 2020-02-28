import configparser
import itertools
import os
import sys

'''
src_net_root_dir = './opt_tfer_expt_series/cifar_100_expts/keras_cifar_net/src_tasks'
trgt_cfg_root_dir = './cfg_dir2/expt_cfg/tfer_expts/opt_tfer_series_prelim_2'

expt_subdir_base = "keras_cifar_net/trgt_tasks_batch_prelim_2/" # Specify SPC
saved_dir_base = 'balin_cifar_100_expts_v' # Specify 0/checkpoints'
saved_iter_base = '_Src_Epochs' # Specify Source Epoch Training Iters
data_loader_base = 'cifar100_trgt_living_vs_notliving_subset_' # Specify SPC sndTrgt Train Set
    
file_name_prefix = 'cifar100_nonliving'
file_name_midfix = '_src_epoch_'
file_name_suffix = '.cfg'


src_net_list = ['7', '8']
spc_list = ['10', '25', '50']
#src_epoch_list = ['0',
#                  '20', '40', '60', '80', '100',
#                  '120', '140', '160', '180', '200',
#                  '225', '250', '275', '300',
#                  '350', '400',
#                  '450', '500',
#                  'best']
#src_epoch_list = ['0',
#                  '25', '50', '75', '100',
#                  '125', '150', '175', '200',
#                  '220', '240', '260', '280', '300',
#                  '320', '340', '360', '380', '400'
#                  '420', '440', '460', '480', '500'
#                  'best']
src_epoch_list = ['400', '420', 
                  '500', 'best']

trgt_train_id_list = ['a', 'b', 'c', 'd', 'e']
tot_list = [src_net_list, spc_list, src_epoch_list, trgt_train_id_list]
'''

def is_int(in_str):
    try:
        int(in_str)
        return True
    except ValueError:
        return False

def make_config():
    config = configparser.ConfigParser()

    config.add_section('ExptFiles')
    config.add_section('NetParams')
    config.add_section('ExptParams')
    config.add_section('SavedParams')
    config.add_section('TrgtTaskParams')

    config['ExptFiles']["data_loader"] = "cifar100_src_living_vs_notliving"
    config['ExptFiles']["class_names"] = "dataset_info/cifar100_dicts_all.pkl"
    config['ExptFiles']["encoding_cfg"] = "cfg_dir2/enc_cfg/softmax_encoding_65.cfg"
    config['ExptFiles']["root_expt_dir"] = "opt_tfer_expt_series_prelim_2"
    config['ExptFiles']["expt_dir"] = "cifar_100_expts"
    config['ExptFiles']["expt_subdir"] = "" # Specify SPC

    config['NetParams']["arch_module"] = "cifar100_keras_net"
    config['NetParams']["output_activation"] = "softmax"
    config['NetParams']["optimizer_cfg"] = "cfg_dir2/opt_cfg/optimizer_tfer_cifar_src_1.cfg"
    config['NetParams']["loss_fnc"] = "categorical_crossentropy"

    config['ExptParams']['epochs'] = '500'
    config['ExptParams']['batch_size'] = '32'
    config['ExptParams']['epochs_per_recording'] = '100'

    config['SavedParams']['saved_set_dir'] = 'opt_tfer_expt_series/cifar_100_expts/keras_cifar_net/src_tasks'
    config['SavedParams']['saved_dir'] = '' # Specify 0/checkpoints'
    config['SavedParams']['saved_arch'] = 'init_arch.json'
    config['SavedParams']['saved_iter'] = '' # Specify Source Epoch Training Iters
    config['SavedParams']['saved_encodings_iter'] = '1'

    config['TrgtTaskParams']['num_reset_layers'] = '1'
    config['TrgtTaskParams']['penultimate_node_list'] = '[]'
    config['TrgtTaskParams']['output_activation'] = 'softmax'
    config['TrgtTaskParams']['data_loader'] = '' # Specify Trgt Train Set

    config['TrgtTaskParams']['class_names'] = 'dataset_info/cifar100_dicts_all.pkl'
    config['TrgtTaskParams']['encoding_cfg'] = 'cfg_dir2/enc_cfg/softmax_encoding_35.cfg'

    return config


def write_cfg_files(config_infile, config_outfile):

    src_net_root_dir =  config_infile['RootDirs']['src_net_root_dir']
    trgt_cfg_root_dir = config_infile['RootDirs']['trgt_cfg_root_dir']

    expt_subdir_base = config_infile['BaseDirStrs']['expt_subdir_base']
    saved_dir_base = config_infile['BaseDirStrs']['saved_dir_base']
    saved_iter_base = config_infile['BaseDirStrs']['saved_iter_base']
    data_loader_base = config_infile['BaseDirStrs']['data_loader_base']
    
    file_name_prefix = config_infile['CfgFileSubstrings']['file_name_prefix']
    file_name_midfix = config_infile['CfgFileSubstrings']['file_name_midfix']
    file_name_suffix = config_infile['CfgFileSubstrings']['file_name_suffix']

    src_net_list = config_infile['ExptParams']['src_net_list'].split(',')
    spc_list = config_infile['ExptParams']['spc_list'].split(',')
    src_epoch_list = config_infile['ExptParams']['src_epoch_list'].split(',')
    trgt_train_id_list = config_infile['ExptParams']['trgt_train_id_list'].split(',')


    src_net_list = [x.strip('[] ') for x in src_net_list] 
    spc_list = [x.strip('[] ') for x in spc_list]
    src_epoch_list = [x.strip('[] ') for x in src_epoch_list]
    trgt_train_id_list = [x.strip('[] ') for x in trgt_train_id_list]
    param_lists = [src_net_list, spc_list, src_epoch_list, trgt_train_id_list]


    
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

    exec_sh_root_dir = config_infile['ExecParams']['exec_sh_root_dir']
    exec_sh_spc_dir = config_infile['ExecParams']['exec_sh_spc_dir']
    exec_sh_file_prefix = config_infile['ExecParams']['exec_sh_file_prefix']
    exec_sh_file_suffix = config_infile['ExecParams']['exec_sh_file_suffix']
    os.makedirs(exec_sh_root_dir, exist_ok=True)

    sh_str = "#!  /bin/bash \n\n"
    cmd_str_root = "python run_expt.py "
    cmd_str_suffix = " > /dev/null 2>&1"

    arg_root_dir = os.path.join(config_infile['RootDirs']['trgt_cfg_root_dir'], "src_net_")
    arg_file_pre_str = config_infile['CfgFileSubstrings']['file_name_prefix'] + \
                       config_infile['CfgFileSubstrings']['file_name_midfix']
    arg_file_end_str = '.cfg --nocheckpoint'

    full_batch_name = 'tfer_net_batch.sh'
    full_batch_path = os.path.join(exec_sh_root_dir, full_batch_name)
    full_batch_file = open(full_batch_path, 'w')
    full_batch_file.write(sh_str)
    echo_str = "echo 'time " + full_batch_path +"' `date` \n"
    full_batch_file.write(echo_str)

    src_net_list = config_infile['ExptParams']['src_net_list'].split(',')
    spc_list = config_infile['ExptParams']['spc_list'].split(',')
    src_epoch_list = config_infile['ExptParams']['src_epoch_list'].split(',')
    trgt_train_id_list = config_infile['ExptParams']['trgt_train_id_list'].split(',')

    src_net_list = [x.strip('[] ') for x in src_net_list] 
    spc_list = [x.strip('[] ') for x in spc_list]
    src_epoch_list = [x.strip('[] ') for x in src_epoch_list]
    trgt_train_id_list = [x.strip('[] ') for x in trgt_train_id_list]

    for curr_SRC_NET in src_net_list:
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

           

if __name__ == "__main__":
    config_root_dir = "./cfg_dir2/gen_cfg/"
    config_infile_name = sys.argv[1]
    
    config_infile = configparser.ConfigParser()
    config_infile.read(os.path.join(config_root_dir, config_infile_name))

    config_outfile = make_config()
    write_cfg_files(config_infile, config_outfile)
    write_shell_scripts(config_infile)
    
    #python make_cfg_files.py opt_tfer_prelim2.cfg
