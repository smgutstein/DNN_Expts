make_tfer_cfg_files.py is used to create the cfg files used to run
a series of tfer expts for different trgt training sets and different
src epochs used for tfer. make_tfer_cfg_files.py will take as its first
argument a path to the cfg file (in local_cfg) that specifes the full
paths to the three cfg fles that will be used to create the cfg files
used in the actual expt.

The cfg fle used directly by make_tfer_cfg_files.py hasthe form:

[CfgPathStrs]
root: ../cfg_dir/gen_cfg/opt_tfer_expts
branch: cifar100_living_living_expts/tfer_datasets
subset_file: subsets.cfg
skel_file: tfer_net_skeleton_0.cfg
var_file: tfer_net_vars_0.cfg

The 3 files specified here are:

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
