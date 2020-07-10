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
