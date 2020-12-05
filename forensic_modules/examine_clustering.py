import warnings
from typing import Any, Union

warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
from collections import OrderedDict, defaultdict
import configparser
from get_class_samples import SamplesCollater
import importlib
import inspect
import keras
from keras.models import model_from_json, model_from_yaml
import math
import numpy as np
import os
import pickle
import sys

def is_number(in_str):
    try:
        float(in_str)
        return True
    except ValueError:
        return False

class ClusterChecker(object):

    def __init__(self, forensics_file, net_id='best_key'):
        self.config = configparser.ConfigParser()
        self.forensics_file = os.path.join('./cfg_dir/net_forensics_cfg',
                                           forensics_file)
        # Get dataset generators
        self.samples = SamplesCollater(self.forensics_file)

        # Get correct encodings (recover_encoding indicates initial net used
        # src task data, but this task is for tfer task data)
        if self.samples.encoding_module == 'recover_encoding':
            self.result_check = True
        else:
            self.result_check = False

        # Get cfg info needed to reconstruct src nets
        self.get_param_dicts()

        # Get info needed to compile net
        self.get_optimizer()
        self.get_metrics()

        # Get net arch
        self.init_nets()

        # Get weights for specific net
        self.get_net(net_id)


    def get_param_dicts(self):
        '''Creates dicts with info from various cfg files'''
        # Get cfg files for net to be examined
        if not os.path.isfile(self.forensics_file):
            print("Can't find %s. Is it a file?" % self.forensics_file)
            sys.exit()
        self.config.read(self.forensics_file)

        # Get path to cfg file for saved expt
        forensics_file_param_dict = self.get_param_dict('SavedNetPathParams')
        self.expt_cfg_file = os.path.join(forensics_file_param_dict['root_dir'],
                                          forensics_file_param_dict['expt_dir'],
                                          forensics_file_param_dict['arch_dir'],
                                          forensics_file_param_dict['net_type'],
                                          forensics_file_param_dict['cfg_file'])

        # Declare path to results file with cluster stats
        self.cluster_stats_file = os.path.join('results',
                                               'cluster_files',
                                               forensics_file_param_dict['result_file']
                                               )

        # Get cfgs for expt run with net, and for net
        if not os.path.isfile(self.expt_cfg_file):
            print("Can't find %s. Is it a file?" % self.expt_cfg_file)
            sys.exit()
        self.config.read(self.expt_cfg_file)

        # Read in cfg info from expt_cfg file
        expt_file_param_dict = self.get_param_dict('ExptFiles')
        net_param_dict = self.get_param_dict('NetParams')

        # Get name of loss function
        self.loss_fnc = net_param_dict['loss_fnc']

        # Get Metric Params
        self.config.read(expt_file_param_dict['encoding_cfg'])
        self.metric_param_dict = self.get_param_dict('MetricParams')

        # Get Optimizer Params (This might not be necessay, since no training will be done)
        self.config.read(net_param_dict['optimizer_cfg'])
        self.optimizer_param_dict = self.get_param_dict('OptimizerParams')

        # Get Location Of Results From Original Experiments
        self.results_root_dir = os.path.join(expt_file_param_dict['root_expt_dir'],
                                             expt_file_param_dict['expt_dir'],
                                             expt_file_param_dict['expt_subdir'],
                                             forensics_file_param_dict['machine_name'])


    def get_param_dict(self, dict_name):
        '''Takes specific info from cfg file and converts to dict'''
        param_dict = {}
        try:
            params = self.config.items(dict_name)
            for curr_pair in params:
                param_dict[curr_pair[0]] = curr_pair[1]
                
            # Convert numeric strings to float
            param_dict = {x:float(param_dict[x])
                               if is_number(param_dict[x])
                                 else param_dict[x]
                               for x in param_dict}

            # Convert non-numeric strings to correct variable types
            for x in param_dict:
                if str(param_dict[x]).lower() == 'none':
                    param_dict[x] = None
                elif str(param_dict[x]).lower() == 'true':
                    param_dict[x] = True
                elif str(param_dict[x]).lower() == 'false':
                    param_dict[x] = False

        except configparser.NoSectionError:
            pass
        return param_dict

    def init_nets(self):
        '''Gets net architecture & stores list of training snapshot names'''
        # Find file specifying net architecture
        init_arch = [x for x in os.listdir(os.path.join(self.results_root_dir,
                                                        'checkpoints'))
                     if 'arch' in x][0]

        # Load net architecture
        arch_file_name = os.path.join(self.results_root_dir,
                                      'checkpoints',
                                      init_arch)
        with open(arch_file_name, 'r') as f:

            print("Loading Architecture from: ", arch_file_name)
            if init_arch[-4:] == 'json':
                json_str = f.read()
                self.model = model_from_json(json_str)
            elif init_arch[-4:] == 'yaml':
                yaml_str = f.read()
                self.model = model_from_yaml(yaml_str)

        # Get list of saved nets and sort by epochs trained
        def net_id(x):
            return int(x.split('_')[-1].split('.')[0])

        net_list = [x for x in os.listdir(os.path.join(self.results_root_dir,
                                               'checkpoints'))
                         if 'weight' in x]
        net_list.sort(key=net_id)
        self.net_dict = OrderedDict()
        for curr in net_list:
            self.net_dict[net_id(curr)] = curr
            if 'best' in curr:
                temp = net_id(curr)
                self.best_epoch = net_id(curr)
        self.net_dict['best_key'] = temp

    def get_net(self, net_id):
        # Load weights for specified net & rebuild it. net_id indicates either
        # the number of training epochs used to create the net, or that this
        # net had the best testing set performance
        self.curr_epoch = self.net_dict[net_id] if net_id == 'best_key' else net_id
        self.get_weights()
        self.build_model()

    def get_metrics(self):
        # Import accuracy function
        temp = importlib.import_module(self.metric_param_dict['metrics_module'])
        self.metric_fnc = getattr(temp, self.metric_param_dict['accuracy_metric'])
        self.metric_fnc_args = inspect.signature(self.metric_fnc)
        
        # NOTE: Need top ensure correct acc fnc is obtained when recovering
        # previous encoding
        if 'y_encode' in self.metric_fnc_args.parameters:
            print("This net used random encoding. This code needs to be modified to handle it")
            sys.exit()
            #metric_fnc = metric_fnc(self.data_manager.encoding_matrix)
            #print("Warning: Need to ensure correct acc", end=' ')
            #print("function is obtained with reuse of encoding")

    def get_optimizer(self):
        # Reload optimization method
        optimizer_module = self.optimizer_param_dict.pop('optimizer_module')
        optimizer = self.optimizer_param_dict.pop('optimizer')
        temp = importlib.import_module(optimizer_module)
        optimizer_fnc = getattr(temp, optimizer)
        self.opt = optimizer_fnc(self.optimizer_param_dict)

    def get_weights(self):
        # Load saved weights into model
        wt_file = os.path.join(self.results_root_dir,
                               'checkpoints',
                               self.net_dict[self.curr_epoch])

        if os.path.isfile(wt_file):
            print("Loading weights from: ", wt_file)
            self.model.load_weights(wt_file)
        else:
            print("Could not find %s. Is it a file ?\n"% wt_file)
            sys.exit()

    def build_model(self):
        '''Compiles both orig model and version used to extract intermediate
           layers.... Not certain orig model needed to be compiled'''
        # Compile model
        print("\nCompiling model ...")
        self.model.compile(loss=self.loss_fnc,
                           optimizer=self.opt,
                           metrics=[self.metric_fnc])

        # Extractor only tracks output layer (in order to verify
        # if input is ultimately classified correctly) and
        # penultimate layer to observe pre-classification clustering
        print("\nCompiling extractor ...")
        self.extractor = keras.Model(inputs=self.model.inputs,
                                     outputs=[layer.output
                                              for layer in self.model.layers][-2:])
        self.extractor.compile(loss=self.loss_fnc,
                           optimizer=self.opt,
                           metrics=[self.metric_fnc])


    def get_class_results(self, curr_class, train=True):
        '''See how well model works for given class (i.e. Recall)'''
        self.samples.get_class_generator(curr_class, train)
        num_samples = self.samples.sub_data.shape[0]
        batches_per_epoch = int(num_samples // self.samples.batch_size) + 1
        score = self.model.evaluate_generator(self.samples.sub_data_gen,
                                              steps=batches_per_epoch)

        return score

    def get_results(self, train=True):
        '''See how well model works for all classes'''
        self.samples.get_full_generators(train)
        num_samples = self.samples.data.shape[0]
        batches_per_epoch = int(num_samples // self.samples.batch_size) + 1
        score = self.model.evaluate_generator(self.samples.data_gen,
                                              steps=batches_per_epoch)
        print ("Score = ", score)
        return score

    def get_class_features(self, curr_class, train=True):
        '''Extract intermediate results for specific class'''
        self.samples.get_class_generator(curr_class, train)
        num_samples = self.samples.sub_data.shape[0]
        batches_per_epoch = int(num_samples // self.samples.batch_size) + 1
        features = self.extractor.predict_generator(self.samples.sub_data_gen,
                                                    steps=batches_per_epoch)
        return features



class ClusterStats(object):

    def __init__(self, forensics_file):
        self.c_checker = ClusterChecker(forensics_file)
        self.net_dict = self.c_checker.net_dict
        self.samples = self.c_checker.samples
        self.results_file = self.c_checker.cluster_stats_file
        self.t_epoch_class_dict = dict()

    def get_vectors(self, epoch, train):
        self.c_checker.get_net(epoch)
        if self.c_checker.result_check:
           self.c_checker.get_results(train)
        vec_dict = dict()
        #print("    ", end="")
        print("Epoch: {:d}  Class Id: ".format(epoch), end = "")
        for class_id in sorted(list(self.samples.data_dict)):
            print("{:d}".format(class_id), end = " ")

            vecs = self.c_checker.get_class_features(class_id, train)
            vec_dict[class_id] = vecs
        print("")
        return vec_dict

    def get_run_data(self):
        '''Create dictionary storing representation of each input
           at given hidden layer (usually LAST hidden layer) for
           all training epochs'''
        epoch_list = sorted([x
                             for x in self.c_checker.net_dict.keys()
                             if isinstance(x, int)])
        self.samples.make_quiet()
        for train_test in [True, False]:
            print("Train ...") if train_test else print("Test...")
            tt_dict = dict()
            for epoch in epoch_list:
                #print("  Epoch: {:d}".format(epoch))
                vec_dict = self.get_vectors(epoch, train_test)
                tt_dict[epoch] = vec_dict
            if train_test:
                self.t_epoch_class_dict['train'] = tt_dict
            else:
                self.t_epoch_class_dict['test'] = tt_dict
        return None

    def save_stats(self, pklfile):
        '''Saves dict with results to pkl file'''
        outpath = self.results_file
        print("Saving ....{:s} ".format(outpath), end=" ... ")
        pickle.dump(self.t_epoch_class_dict,
                    open(outpath,'wb'))
        print("Done")

if __name__ == '__main__':

    # Get config file from cmd line
    parser = argparse.ArgumentParser(
        description="Run Keras Expt With Specified Output Encoding")
    parser.add_argument('forensics_file', action='store',
                        type=str, nargs='*', default='')

    cmd_line_args = parser.parse_args()
    forensics_file = cmd_line_args.forensics_file[0]

    # Load initial data, saved nets and prepare output file
    x = ClusterStats(forensics_file)

    # Gets hidden representations for all data at all snapshots
    x.get_run_data()

    # Saves data for later analysis
    x.save_stats('test_cluster_tfer.pkl')


