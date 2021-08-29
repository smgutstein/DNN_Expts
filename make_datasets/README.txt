Files in this directory exist to make datasets useable for transfer expts.
First, they split datasets into source and target datasets. It will also make smaller subsets
of target datasets, to reduce them to a specified number of samples per class (SPC).
Finally, it will create the modules needed to load the resulting datasets.

1. Divide dataset into src and trgt tasks:
python make_src_trgt_task ./local_cfg/make_cifar100_src_trgt_tasks_0.cfg

./local_cfg/make_cifar100_src_trgt_tasks_0.cfg is a cfg file that gives path to cfg file
that specifies how classes are to be split, and path to raw data. It has the following
form:

[CfgPathStrs]
root: ..
branch: dataset_info
leaf: cifar100_src_trgt_v1.cfg

[RawDataPathStrs]
root: .keras
branch: datasets
leaf: cifar-100-python

It should be noted that we're assuming datais stored in format that CIFAR data is stored
in at download. So, a sample file indicating the sourcetask/target task division has the
followng form:

[MetaData]
class_type: coarse
samps_per_class_training: 2500
samps_per_class_testing: 500

[Tasks]
source: aquatic_mammals, fish, flowers, fruit_and_vegetables, insects, large_carnivores,
        large_omnivores_and_herbivores,
        medium_mammals, non-insect_invertebrates, people, reptiles,
	small_mammals, trees

target: food_containers, household_electrical_devices, household_furniture,
        large_man-made_outdoor_things, large_natural_outdoor_scenes, vehicles_1,
	vehicles_2

[output_names]
expt: cifar100_living_notliving
source: Animals_Plants
target: Non_Living_Things

2. Create smaller trgt class datasets with specified SPC:

python make_spc_training_sets.py ./local_cfg/make_cifar100_spc_training_sets_0.cfg

make_cifar100_spc_training_sets_0.cfg specifies which cfg file is used to define subset, and the
name of the module that will be used to load data. It has the following format:

[PathStrs]
root: ../cfg_dir/gen_cfg/opt_tfer_expts
branch: cifar_100_living_notliving_expts/tfer_datasets
leaf: subsets.cfg
load_module: cifar

The file (in this case - ../cfg_dir/gen_cfg/opt_tfer_expts/cifar_100_living_notliving_expts/tfer_datasets/subsets.cfg)
used to define those subsets has the following format:

[Subsets]
spc: 10
suffixes: a,b,c,d,e

[StorageDirectory]
data_root_dir: .keras/datasets
data_dir: cifar-100-python
subset_root_dir: cifar100_living_notliving
subset_dir: trgt_tasks

[Notes]
note: CIFAR 100 living vs not living datasets

It specifies how many SPC areto be used (I believe it can also handle lists - i.e. spc: 5,10),
the suffixes to be given to each different trgt task dataset, and it gives a path to
the directory where these datasets will be stored.

3. Create modules used to load data

python make_trgt_dataloaders.py ./local_cfg/make_cifar100_spc_training_sets_0.cfg

This is the same cfg file as used before. Possibly, make_spc_training_sets and make_trgt_dataloaders
should be combined.



========================================================================================================================================================
OLD NOTES:

Some notes on files in this directory:

make_cifar_src_trgt_tasks.py - Used to split CIFAR-100 into source task and target task subsets. Uses a cfg file, which is given as 1st cmd line argument

Sample cfg file for make_cifar_src_trgt_tasks:
[MetaData]
class_type: coarse
samps_per_class_training: 2500
samps_per_class_testing: 500

[Tasks]
source: aquatic_mammals, fish, flowers, fruit_and_vegetables, insects, large_carnivores,
        large_omnivores_and_herbivores,
        medium_mammals, non-insect_invertebrates, people, reptiles,
	small_mammals, trees

target: food_containers, household_electrical_devices, household_furniture,
        large_man-made_outdoor_things, large_natural_outdoor_scenes, vehicles_1,
	vehicles_2

[output_names]
expt: Living_vs_Not_Living
source: Animals_Plants
target: Non_Living_Things

------------------------------------------------------------------
make_cifar_spc_training_sets.py - Used to create training sets for target task with requisite number of samples per class (spc). Uses cfg file,  which is given as 1st cmd line argument

Sample cfg file for make_cifar_spc_training_sets (also used by make_cifar_trgt_dataloaders.py):

[Subsets]
spc: 50
suffixes: a,b,c

[StorageDirectory]
data_root_dir: .keras/datasets
data_dir: cifar-100-python
subset_root_dir: cifar100_living_living
subset_dir: trgt_tasks

[Notes]
note: CIFAR 100 living vs living datasets
------------------------------------------------------------------
make_cifar_trgt_dataloaders.py - Used to create dataset_loaders for various target task training datasets. Uses cfg file,  which is given as 1st cmd line argument

Sample cfg file for make_cifar_spc_training_sets (also used by make_cifar_spc_training_sets.py):

[Subsets]
spc: 50
suffixes: a,b,c

[StorageDirectory]
data_root_dir: .keras/datasets
data_dir: cifar-100-python
subset_root_dir: cifar100_living_living
subset_dir: trgt_tasks

[Notes]
note: CIFAR 100 living vs living datasets
----------------------------------------------------------------
trgt_task_dataloader_strings.py - Stores static strings used to create target task dataset_loaders. Is imported by make_cifar_trgt_dataloaders.py
----------------------------------------------------------------
