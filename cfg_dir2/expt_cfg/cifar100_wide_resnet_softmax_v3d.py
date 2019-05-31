[ExptFiles]
data_loader: cifar100
class_names: dataset_info/cifar100_dicts_all.pkl
encoding_cfg: cfg_dir2/enc_cfg/softmax_encoding_100.cfg
root_expt_dir: encoding_expts_3
expt_dir: cifar_100_expts
expt_subdir: wide_resnet_28_10

[NetParams]
arch_module: wide_resnet_28_10
output_activation: softmax
optimizer_cfg: cfg_dir2/opt_cfg/optimizer_sgd3.cfg
loss_fnc: categorical_crossentropy

[ExptParams]
epochs: 500
batch_size: 128
epochs_per_recording: 50
