from __future__ import print_function
import argparse
import ConfigParser
import datetime
import importlib
import os

from cifar_10 import Cifar_Net
from data_manager_recon_cifar10 import DataManager



if __name__ == '__main__':
    
    start_time = datetime.datetime.now()

    # Get config file from cmd line
    parser = argparse.ArgumentParser(
        description="Run Keras Expt With Specified Output Encoding")
    parser.add_argument('config_file', action='store', type=str, default = '')
    args=parser.parse_args()

    # Get input args from config file
    if not os.path.isfile(args.config_file):
        print("Can't find %s. Is it a file?"%(args.config_file))
        os._exit(1)
        
    config = ConfigParser.ConfigParser()
    config.read(args.config_file)

    # Get Encoding Parameters
    encoding_module = config.get('Encoding', 'EncodingModule')
    nb_code_bits = config.getint('Encoding', 'nb_code_bits')
    encoding_params = config.items('EncodingModuleParams')
    encoding_param_dict = {}
    for curr_pair in encoding_params:
        encoding_param_dict[curr_pair[0]] = curr_pair[1]

    # Get Expt Parameters
    epochs = config.getint('ExptParams','epochs')
    metrics_module = config.get('ExptParams','metrics_module')
    accuracy_metric = config.get('ExptParams','accuracy_metric')
    temp = importlib.import_module(metrics_module)
    acc_fnc = getattr(temp, accuracy_metric)
    

    x = DataManager(encoding_module, nb_code_bits)
    x.make_encoding_dict(**encoding_param_dict)
    x.encode_labels()
    y = Cifar_Net(epochs, x, 'temp', 'expt1', acc_fnc)
    y.train()
    

        
