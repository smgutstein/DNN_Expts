Starting Batch Experiments:

python make_cfg_and_sh_files.py <name of cfg file>

make_cfg_and_sh_files.py constructs a tree of shell scripts used to run a batch of experiments. Each shell script calls each of its children shell scripts until a `leaf' shell script is reached, which will actually run a set number of experiments. The tree is traversed in depth-first fashion.

A sample cfg file used by  make_cfg_and_sh_files.py, looks like this:

[RootDirs]
src_net_root_dir: <root directory with various source task nets>
trgt_cfg_root_dir: <root directory with cfg files for each trgt task expt>

[BaseDirStrs]
expt_subdir_base: <base subdirectory where expt results are stored. Due to ad-hoc planning, this directory specifies the architecure used, with a batch identifier as a sub-dir - e.g. res_net_100/trgt_tasks_batch_2. The parent dir specifies the dtaset being used for the expts, and its parent is trgt_cfg_root_dir>
saved_dir_base: <leaf-ish sub-directory where results for a single expt are stored> 
saved_iter_base: <"_Src_Epochs" - this shouldn't be dynamic input. It's a suffix for a sub-dir name indicating number of training epochs on source task for all expts in child dirs> 
data_loader_base: <prefix for dir with datset loader. suffix is generally 'a', 'b', 'c' etc. indicating which instantiation of data is being used>

[CfgFileSubstrings]
file_name_prefix: cifar100_nonliving
file_name_midfix: _src_epoch_
file_name_suffix: .cfg

[ExptParams]
src_net_list: <comma separated list of source nets>
spc_list: <comma separated list of samples per class for trgt classes>
src_epoch_list: <comma separated list of training epochs on source task. May be either number, or 'best', for training epoch with best performance on validation set>
trgt_train_id_list: <comma separated list of training set ids for target set>

[ExecParams]
exec_sh_root_dir: <directory where shell scripts for batch expts will be stored>

==========================================================================================

Run An Individual Experiment

python run_expt.py <name of cfg file>

run_expt.py is the base code used to run each individual experiment. The batch file tree, described above, eventually terminates in a shell script that executes individual expts using run_expt.py

[ExptFiles]
data_loader = <file within dataset_loaders dir that loads datasets for source task of this expt>
class_names = <path to pkl file that maps class numbers to names and vice-versa for source task>
encoding_cfg = <path to cfg file containing info for encoding classes of source task>
root_expt_dir = <root dir for expts>
expt_dir = <subdir for all expts with given dataset>
expt_subdir = <subdir specifying further details - i.e. architecture/name of subset of expts/name of source task net/samples per class (trgt data)>

[NetParams]
arch_module = <module specifying how to build architecture. Stored in net_architectures sub-dir>
output_activation = <activation function at output layer>
optimizer_cfg = <cfg file for optimization technique>
loss_fnc = <loss function>

[ExptParams]
epochs = <# of trining epochs>
batch_size = <# data samples per batch>
epochs_per_recording = <# of epochs between saved nets>

[SavedParams]
saved_set_dir = <root dir for source net expts>
saved_dir = <sub-dir with saved checkpoints>
saved_arch = <file specifying source net architecture>
saved_iter = <string of form # of Source Epochs  '_Src_Epochs' - (i.e. 420_Src_Epochs)
saved_encodings_iter = <iteration with final encoding specification - usually(always) 1>

[TrgtTaskParams]
num_reset_layers = <number of layers in source net to be reset for target task - usually(always) 1>
penultimate_node_list = <An empty list - [], I don't recall what I was thinking here>
output_activation = <output activation function for trgt task net>
data_loader = <file within dataset_loaders dir that loads datasets for trgt task of this expt>
class_names = <path to pkl file that maps class numbers to names and vice-versa for trgt task>
encoding_cfg = <path to cfg file containing info for encoding classes of target task>
==========================================================================================================
encoding cfg - cfg file for encoding method. Stored in cfg_dir2/enc_cfg

Softmax Encoding

[Encoding]
#These params are needed
#by any encoding
nb_code_bits: <Number of Output Nodes Used To Encode Classifications>
hot_not_hot_fnc: <Function returning hot/not hot values for output nodes stored in hot_not_hot_values.py>
hot_val: <Repetitive input - Try not to conflict with function above>
not_hot_val: <Repetitive input - Try not to conflict with function above>
saved_encoding: <File with saved encoding from previous run - need to research how it's used in code>


[EncodingModuleParams]
#These params are specific to
#the encoding method
encoding_module: <module assigning output codes to classes - in this case: n_hot_encoding>
nb_hot: <number of `hot' bits per class - in this case - 1>

[MetricParams]
metrics_module: <module containing metric functions - local_metric>
accuracy_metric: <function from metrics module - in this case: hot_bit>
...............................................................................................
Random Encoding
[Encoding]
#These params are needed
#by any encoding
nb_code_bits: <Number of Output Nodes Used To Encode Classifications>
hot_not_hot_fnc: <Function returning hot/not hot values for output nodes stored in hot_not_hot_values.py>

[EncodingModuleParams]
#These params are specific to
#the encoding method
encoding_module:  <module assigning output codes to classes - in this case: rnd_encoding>
seed: <random seed used for rnd encodings - in this case: 1453>
hot_prob: <probability of an output code bit being assigned hot value - in this case - 0.50>

[MetricParams]
metrics_module:  <module containing metric functions - local_metric> 
accuracy_metric:  <function from metrics module - in this case: ECOC_top_1>
-------------------------------------------------------------------------------------------------
optimization cfg - cfg file for optimization method. Stored in cfg_dir2/opt_cfg

[OptimizerParams]
optimizer_module: <file with optimizer funstion - usually local_optimizer>
optimizer: <optimizer function - in this case: sgd_schedule>
lr_schedule: <If sgd_schedule is chosen optimizer method, then a this param is a series of ordered pairs, separated by commas. The first element of each ordered pair is an epoch number. The second element is the learning rate that is first initiated at that epoch - e.g. (0,0.001), (5, .0005), (10, .0001), (20, .00005)>
decay: <decay rate of learning rate>
momentum: <momentum>
nesterov: <boolean indicating if Nesterov acceleration is used>

===================================================================================================

Package Structure:

Root Directory:
cfg_dir: Old directory of cfg files

cfg_dir2: New directory of cfg files

dataset_info: Directory with files with info about datasets:
   -- pkl files with dicts mapping class numbers to names
   -- pkl files mapping cifar 100 coarse classes to fine classes
   -- cfg file splitting single dataset into source & target datasets
   
display_data: Files used to display datasets whether stored in hdf format or large file:
  -- display_cifar_subsets.py & display_cifar_subsets_hdf5.py : Get cifar data whether stored in std. file or hdf5 file
  -- data_display.py, data_display_hdf5.py : Display data

execute_expts: Directory with shell scripts used to run indiv expts and batches of expts. Also contains dirctory trees with shell script trees for batch expts.

external dir: Directory with files from PyImageSearch

hdf_builder: Directory with files to build datasets stored in hdf5 format
  The files in this directory seem to be pretty much just run "as is". They take a specific dataset, as indicated in their name and save it in hdf format. There is a config sub-directory, which has config data used to make the conversions. Likely, these files could be replaced with some sort of generalized file that would work for any dataset.

local_backend: Directory with Theano and Tensorflow backends. Originally, this package was meant to be usable with both Theano and Tensorflow, but since Theano is no longer maintained. The local_tensorflow_backend.py file has all standard backend functions, along with some extra written with latent learning in mind - i.e. classification that doesn't assume 1-of-n encoding. Also, functions for "Top-N" accuracy.

make_datasets: Files used to create datasets and dataloaders. Again, most of these files seem to be specially designed for a single dataset.

net_architectures: Directory with files used to construct different architectures. These files all contain a sincle function called "build_arcitecture", which takes 3 arguments: 1. input_shape = (channels, rows, cols) 2. nb_output_nodes = number of output node (remember - 1-of-N encoding is not assumed, so number of output nodes might not equal number of classes) 3. Output activation function.

This directory also contains sg_activation.py - a modification of keras' Activation class, which names each actiivation layer after its speific activation function, rather than with the generic name "Activation".

notebooks: Directory with all Jupyter notebooks. It is split into 2 sub-directories - "Save" and "Scrap". Save is meant to contain publishable/published results. "Scrap" is meant to contain everything else.

ALL Other Dirs were used to stored expt results.
