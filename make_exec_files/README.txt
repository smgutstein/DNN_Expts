make_tfer_exec_files.py is used to create exec files for series of tfer expts.
It uses a cfg file name in local_cfg as first argument to get path
to actual cfg_file used to make exec files. It generally has the format:

[PathStrs]
root: ../cfg_dir/gen_cfg/opt_tfer_expts
branch: cifar100_living_living_expts
sub_branch: wide_resnet_28_10_arch
leaf: tfer_nets
exec_file: exec.cfg

The exec file has the format:

 [DirStructure]
    expt_class: opt_tfer_expts
    expt_datasets: caltech101_living_notliving_expts
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
    src_net_list: [alt01.arl.army.mil_v0,  alt03.arl.army.mil_v0,  alt05.arl.army.mil_v0, alt06.arl.army.mil_v0,  alt09.arl.army.mil_v0 ] #These are names of machines src tasks were run upon. 
    spc_list: 10                                                                                                                          #they are also names of dirs src machines are stored in
    src_epoch_list: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, best
    trgt_train_id_list: a, b, c, d, e

    [ExecParams]
    exec_sh_spc_dir: spc
    exec_sh_file_prefix: caltech101_notliving_trgt_series
    exec_sh_file_suffix: .sh



