This is an ongoing project to create a framework to easily perform experiments on deep neural nets. Most components are fairly stable, and in approximate final form. Some are/should be deprecated and removed. Some have been quickly added, in order to answer specific questions and may/may not become part of the full project.


Running Experiments:

A single experiment is run using run_expt.py as follows:

$ python run_expt.py ./cfg_dir/expt_cfg/<path to cfg file>

To run multiple experiments, shell scripts are created and placed in the execute_expts sub-dir. The path to the shell scripts contains the same information as the paths to the cfg files used in the expts. These multiple expts are run just by executing the shell script:

$ ./execute_expts/<path to shell script>

These shell scripts may be created by hand, or in the cases with large numbers of expts to run over multiple machines, by using the files in make_exec_files to create a cascading series of shell scripts that will run all specified expts, or various subsets of them.

Files Defining Experiments:

Each experiment is specified with a collection of cfg files, which are stored in cfg_dir. This dir is composed of 4 main sub-dirs:

1. enc_cfg - contains files used to specify output encodings
2. expt_cfg - contains files used to specify expt. parameters and other cfg files to be used in given expt
3. gen_cfg - contains files used to generate subsets of datasets. Primarily used to create small training sets with limited samples per class, and to divide a particular dataset into source and target tasks for transfer learning expts.
4. net_forensics_cfg - contains files used to examine state of net during various training epochs - may be deprecated
5. opt_cfg - contains files used to specify optimization parameters

The various paths with each of these directories specify the:
1. series of expts being run
2. dataset being used
3. net architecture being used
4. specific information about net - e.g. is it being used for a source task, target task (transfer) or a lottery ticket hypothesis test.


Creating Experiments:

The four main sub-dirs containing the files used to create the cfg files for a given expt or series of expts are:

1. make_cfg_files - used to create cfg files that define expt or series of expts
2. make_datasets - used to create datasets used for expts. Was generally focused on transfer learning expts, so has files used to divide given dataset into source & target tasks, create target task training sets with specified number of samples per class, and to create dataset_loader modules that are dynamically loaded at run time.
3. make_exec_files - used to create cacading series of shell scripts for performing large number of simulataneous experiments. Cascading structure is tree-like, so it is easy to distribute sets of experiments across multiple machines


Main Files:

1. run_expt.py - Used to run initiate individula expts.
2. data_manager.py - Contains objects used to manage datasets
3. net_manager.py - Contains code used to manage nets

Other Sub-dirs:

1. dataset_info - Contains files with metadata for datasets.
2. dataset_loaders - contains files with "load_data" method, which is dynamically loaded in order to read each specific dataset
3. display_data - code used to directly view datasewts
4. err_logs - directory with stdout & stderr for each expt. - Useful for batch runs
5. external_dir - code 'borrowed' from PyImageSearch
6. forensic_modules - files used to study evolution of nets during training
7. hdf_buildef - files used to store data in hdf format - deprecated
8. local_backend - modification/additions to backend functions
9. lr_sched_fncs - files with learning rate schedule functions
10. net_architectures - files with modules defining net architectures. These aree dynamically loaded at run time.
11. notebooks - Jupyter notebooks used to analyze/graph results
12. results - dir into which all results are stored. Structure of sub-dirs mirrors that in cfg_dir. The path to a given result also gives info about specific expt that was run.




