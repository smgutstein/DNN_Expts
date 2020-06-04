Some notes on files in this directory:

Splitting CIFAR-100 into suubsets (by class) in order to create source and target tasks:
Run make_cifar_src_trgt_tasks.py  -- e.g.  python make_cifar_src_trgt_tasks.py ../dataset_info/cifar100_src_trgt_v1.cfg

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

Creating training sets with set number of samples per class (spc):

python make_cifar_subsets_batch.py

Currently this code automatically goes to a config file (../cfg_dir/gen_cfg/opt_tfer_expts/tfer_datasets/subsets.cfg
to find out how many samples per class the transfer training sets will contain, and how many new training sets are desired. The config file looks like this:

[Subsets]
spc: 1,5
suffixes: a,b

spc is a comma separated list of the number of samples per class in each new training set. suffixes is a comma separated list of the suffixes to be added to each training set's name, in order to differentiate them. 
