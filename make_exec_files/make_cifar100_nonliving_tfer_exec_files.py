import argparse
import configparser
from itertools import product
import os
import sys

def write_shell_scripts(config_infile):
    # Write nested shell scripts used to run full batch of expts

    # Get dirs in dir tree
    expt_class = config_infile['DirStructure']['expt_class']
    expt_datasets = config_infile['DirStructure']['expt_datasets']
    expt_arch = config_infile['DirStructure']['expt_arch']
    spc_prefix = config_infile['DirStructure']['spc_prefix']
    src_epoch_prefix = config_infile['DirStructure']['src_epoch_prefix']
    src_prefix = config_infile['DirStructure']['src_prefix']
    tfer_tr_set_prefix = config_infile['DirStructure']['tfer_tr_set_prefix']
    res_dir = config_infile['DirStructure']['res_dir']
    enc_dir = config_infile['DirStructure']['enc_dir']
    expt_dir = config_infile['DirStructure']['expt_dir']
    src_nets_dir = config_infile['DirStructure']['src_nets_dir']

    # Get data for shell script dir
    exec_sh_root_dir = os.path.join('..','execute_expts', expt_class,
                                    expt_datasets, expt_arch,
                                    'tfer_nets')
    exec_sh_spc_dir = config_infile['ExecParams']['exec_sh_spc_dir']
    exec_sh_file_prefix = config_infile['ExecParams']['exec_sh_file_prefix']
    exec_sh_file_suffix = config_infile['ExecParams']['exec_sh_file_suffix']

    # If necessary make shell script dir
    os.makedirs(exec_sh_root_dir, exist_ok=True)

    # Create cmd strings to be exectued by shell scripts
    sh_str = "#!  /bin/bash \n\n"
    cmd_str_root = "python run_expt.py "
    cmd_str_args = " --nocheckpoint"
    cmd_str_suffix = " > /dev/null 2>&1"

    # Create arguments for cmd strings (i.e. arguments for run_expt.py)
    arg_root_dir = os.path.join('./cfg_dir/expt_cfg',
                                expt_class, expt_datasets,
                                expt_arch, 'tfer_nets')
    arg_file_end_str = '.cfg --nocheckpoint'

    # Create root shell script file
    full_batch_name = 'tfer_net_batch.sh'
    full_batch_path = os.path.join(exec_sh_root_dir, full_batch_name)
    full_batch_file = open(full_batch_path, 'w')
    full_batch_file.write(sh_str)
    echo_str = "echo 'time " + full_batch_path[1:] +"' `date` \n"
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

    for curr_SPC in spc_list:
        # Create shell scripts for give (source net, samples per class)
        curr_spc_dir = os.path.join(exec_sh_root_dir, "spc_" + curr_SPC)
        os.makedirs(curr_spc_dir, exist_ok=True)

        spc_batch_name = 'batch_spc_' + curr_SPC + '.sh'
        spc_batch_path = os.path.join(curr_spc_dir, spc_batch_name)
        print (spc_batch_path)
        
        spc_batch_file = open(spc_batch_path, 'w')
        spc_batch_file.write(sh_str)
        echo_str = "echo '" + spc_batch_path[1:] +"' `date` \n\n"
        spc_batch_file.write(echo_str)

    

        for curr_SRC_EPOCH in src_epoch_list:
            # Create shell scripts for give (source net, samples per class,
            #                                source task training epochs)
            curr_src_epoch_dir = os.path.join(curr_spc_dir, 'src_epoch_' + curr_SRC_EPOCH)
            os.makedirs(curr_src_epoch_dir, exist_ok=True)

            src_epoch_batch_name = 'batch_src_epoch_' + curr_SRC_EPOCH + '.sh'
            src_epoch_batch_path = os.path.join(curr_src_epoch_dir, 
                                                src_epoch_batch_name)
            print ("   ", src_epoch_batch_name)
            
            src_epoch_batch_file = open(src_epoch_batch_path, 'w')
            src_epoch_batch_file.write(sh_str)
            echo_str = "echo '  " + src_epoch_batch_name +"' `date` \n\n"
            src_epoch_batch_file.write(echo_str)

            arg_str_prefix = os.path.join(arg_root_dir + curr_SPC, 
                                          curr_SPC + 'spc')


            for curr_SRC_NET in src_net_list:
                # Create shell scripts for given source net
                machine_name = curr_SRC_NET.split('_')[0]
                version = curr_SRC_NET.split('_')[-1]
                curr_src_net_dir = os.path.join(curr_src_epoch_dir,
                                                '_'.join(['src_net',
                                                          machine_name,
                                                          version]))
                os.makedirs(curr_src_net_dir, exist_ok=True)

                src_net_batch_name = '_'.join(['batch_src_net', machine_name, version + '.sh'])
                src_net_batch_path = os.path.join(curr_src_net_dir, src_net_batch_name)
                print ("      ",src_net_batch_name)
                
                src_net_batch_file = open(src_net_batch_path, 'w')
                src_net_batch_file.write(sh_str)
                echo_str = "echo '    " + src_net_batch_name +"' `date` \n\n"
                src_net_batch_file.write(echo_str)

                for curr_TR_ID  in trgt_train_id_list:
                    # Create shell scripts for give (source net, samples per class,
                    #                                source task training epochs,
                    #                                target task training set)
                    # NOTE: This innermost loop generates cmds for leaf shell script
                    echo_str = "echo '          spc_" + curr_SPC
                    echo_str += "_" + 'src_epoch_' + curr_SRC_EPOCH
                    echo_str += "_tr_set_" + curr_TR_ID +"'  `date` \n"
                    src_net_batch_file.write(echo_str)

                    cfg_file_path = os.path.join(arg_root_dir,
                                                 "spc_" + curr_SPC,
                                                 "src_epoch_" + curr_SRC_EPOCH,
                                                 '_'.join(["src_net",
                                                           machine_name,
                                                           version]),
                                                 "tr_set_" + curr_TR_ID,
                                                 "tfer_net_0.cfg")
                    
                    cmd_str = ' '.join([cmd_str_root, cfg_file_path,
                                        cmd_str_args, cmd_str_suffix, '\n\n'])
                    src_net_batch_file.write(cmd_str)

                src_net_batch_file.close()
                os.chmod(src_net_batch_path, 0o755)
                src_epoch_batch_file.write(src_net_batch_path[1:] + '\n')

            src_epoch_batch_file.close()
            os.chmod(src_epoch_batch_path, 0o755)
            spc_batch_file.write(src_epoch_batch_path[1:] + '\n')

        spc_batch_file.close()
        os.chmod(spc_batch_path, 0o755)
        full_batch_file.write(spc_batch_path[1:] + '\n')

    full_batch_file.close()
    os.chmod(full_batch_path, 0o755)

if __name__ == "__main__":
    config_root_dir = "../cfg_dir/gen_cfg/"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("ExecCfg", help="execute cfg file for series of expts")
    parser.add_argument("--Major", type=str, default = "opt_tfer_expts",
               help="Directory for all expts in series")
    parser.add_argument("--Data", type=str, default = "cifar_100_living_notliving_expts",
               help="Directory for all expts using given datasets")
    parser.add_argument("--Arch", type=str, default = "wide_resnet_28_10_arch",
               help="Directory for all expts using given net architecture")
    parser.add_argument("--Src", action="store_true",
               help="Directory for all expts using given net architecture")
    
    args = parser.parse_args()
    exec_cfg = args.ExecCfg
    major_expts = args.Major
    dataset = args.Data
    arch = args.Arch
    src_nets = args.Src
    if src_nets:
       config_leaf_dir = "src_nets"
    else:
       config_leaf_dir = "tfer_nets"

    exec_cfg_file = os.path.join(config_root_dir, major_expts,
                                 dataset, arch, config_leaf_dir,
                                 exec_cfg)

    print("Reading: ",exec_cfg_file) 
    exec_config = configparser.ConfigParser()
    exec_config.read(exec_cfg_file)
    write_shell_scripts(exec_config)

    '''
    Creates tree of shell scripts to run batches of experiments.
    Uses info from cfg file of this format:

    [DirStructure]
    expt_class: opt_tfer_expts
    expt_datasets: cifar_100_living_notliving_2_expts
    expt_arch: wide_resnet_28_10_arch
    spc_prefix: spc_
    src_epoch_prefix: src_epoch_
    src_prefix: src_net_
    tfer_tr_set_prefix: tr_set_
    res_dir: results
    enc_dir: cfg_dir/enc_cfg
    expt_dir: cfg_dir/expt_cfg
    src_nets_dir: src_nets/workshop_expts

    [ExptParams]
    src_net_list: Chanticleer_cifar_100_living_notliving_expts_v2
    spc_list: 1, 5, 10, 25, 50, 100, 200, 250
    src_epoch_list: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, best
    trgt_train_id_list: a

    [ExecParams]
    exec_sh_spc_dir: spc
    exec_sh_file_prefix: cifar100_living_trgt_series
    exec_sh_file_suffix: .sh
    '''
