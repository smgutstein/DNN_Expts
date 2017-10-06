from __future__ import print_function
import argparse
import ConfigParser
import datetime
import importlib
import os

from cifar_10 import Cifar_Net
from data_manager_recon_cifar10 import DataManager

def get_cmd_line_args():
    
    # Get config file from cmd line
    parser = argparse.ArgumentParser(
        description="Run Keras Expt With Specified Output Encoding")
    parser.add_argument('config_file', action='store', type=str, default = '')
    args=parser.parse_args()

    # Get input args from config file
    if not os.path.isfile(args.config_file):
        print("Can't find %s. Is it a file?"%(args.config_file))
        os._exit(1)

    return args

def get_config_file_params(expt_file):
    
    config = ConfigParser.ConfigParser()
    config.read(expt_file)

    # Get Encoding Parameters
    encoding_params = config.items('Encoding')
    encoding_param_dict = {}
    for curr_pair in encoding_params:
        encoding_param_dict[curr_pair[0]] = curr_pair[1]
    
    # Get kwargs that are specific to
    encoding_module_params = config.items('EncodingModuleParams')
    encoding_module_param_dict = {}
    for curr_pair in encoding_module_params:
        encoding_module_param_dict[curr_pair[0]] = curr_pair[1]

    # Get Expt Parameters
    expt_params = config.items('ExptParams')
    expt_param_dict = {}
    for curr_pair in expt_params:
        expt_param_dict[curr_pair[0]] = curr_pair[1]
    
    # Get Archicture Module
    net_params = config.items('NetParams')
    net_param_dict = {}
    for curr_pair in net_params:
        net_param_dict[curr_pair[0]] = curr_pair[1]


    return[ encoding_param_dict,
            expt_param_dict,
            encoding_module_param_dict,
            net_param_dict]

def run_expt(expt_file):

    # Read cfg file params
    config_params = get_config_file_params(expt_file)
    
    [encoding_param_dict,
     expt_param_dict,
     encoding_module_param_dict,
     net_param_dict]  = config_params

    # Run Expt
    start_time = datetime.datetime.now()
    dm = DataManager(net_param_dict['output_activation'],
                     encoding_param_dict,
                     encoding_module_param_dict)
    net = Cifar_Net(dm, 'temp', 'expt1', net_param_dict, expt_param_dict)
    net.train()
    stop_time = datetime.datetime.now()

    # Show run time (by wall clock)
    run_time = datetime.timedelta
    seconds = int(round(run_time.total_seconds(stop_time - start_time)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    print("Start Time: %s"%(start_time.strftime("%H:%M:%S %p %A %Y-%m-%d")))
    print("Stop Time : %s"%(stop_time.strftime("%H:%M:%S %p %A %Y-%m-%d")))
    print("Run Time  : {:d}:{:02d}:{:02d}".format(hours, minutes, seconds))

    return [dm, net]

        
    
if __name__ == '__main__':

    #Get cmd line args
    args = get_cmd_line_args()
    expt_file = args.config_file
    [dm, net] = run_expt(expt_file)
    import pdb
    pdb.set_trace()
    

    ## # Read cfg file params
    ## config_params = get_config_file_params(args)
    
    ## [encoding_param_dict,
    ##  expt_param_dict,
    ##  encoding_module_param_dict,
    ##  net_param_dict]  = config_params

    ## # Run Expt
    ## start_time = datetime.datetime.now()
    ## dm = DataManager(encoding_param_dict,
    ##                 encoding_module_param_dict)
    ## y = Cifar_Net(dm, 'temp', 'expt1', net_param_dict, expt_param_dict)
    ## y.train()
    ## stop_time = datetime.datetime.now()

    ## # Show run time (by wall clock)
    ## run_time = datetime.timedelta
    ## seconds = int(round(run_time.total_seconds(stop_time - start_time)))
    ## minutes, seconds = divmod(seconds, 60)
    ## hours, minutes = divmod(minutes, 60)
    ## print("Start Time: %s"%(start_time.strftime("%H:%M:%S %p %A %Y-%m-%d")))
    ## print("Stop Time : %s"%(stop_time.strftime("%H:%M:%S %p %A %Y-%m-%d")))
    ## print("Run Time  : {:d}:{:02d}:{:02d}".format(hours, minutes, seconds))

    ## import pdb
    ## pdb.set_trace()
        
