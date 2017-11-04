from __future__ import print_function
import argparse
import ConfigParser
import datetime
import errno
import os
import shutil
import sys

class Logger(object):
    def __init__(self, filename="Expt_output.log"):
        self.filename = filename
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stderr = sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def stop_log(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.terminal = open(os.devnull,'w')
        
    def close_log(self, log_dir = ''):
        self.log.close()
        if log_dir != '':
            shutil.move(self.filename,
                            os.path.join(log_dir, self.filename))

# Capture output with theano/keras & gpu info
expt_log = Logger()
import keras
from cifar_10 import Cifar_Net
from data_manager_recon_cifar10 import DataManager

def get_cmd_line_args():
    
    # Get config file from cmd line
    parser = argparse.ArgumentParser(
        description="Run Keras Expt With Specified Output Encoding")
    parser.add_argument('config_file', action='store', type=str, default='')
    args = parser.parse_args()

    # Get input args from config file
    if not os.path.isfile(args.config_file):
        print("Can't find %s. Is it a file?" % args.config_file)
        os._exit(1)

    return args

def get_encoding_params(expt_file):
    
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
    metric_params = config.items('MetricParams')
    metric_param_dict = {}
    for curr_pair in metric_params:
        metric_param_dict[curr_pair[0]] = curr_pair[1]

    return[encoding_param_dict,
           encoding_module_param_dict,
           metric_param_dict]

def get_expt_params(expt_file):

    config = ConfigParser.ConfigParser()
    config.read(expt_file)

    # Get file info
    file_params = config.items('ExptFiles')
    file_param_dict = {}
    for curr_pair in file_params:
        file_param_dict[curr_pair[0]] = curr_pair[1]

    # Get Architecture Module
    net_params = config.items('NetParams')
    net_param_dict = {}
    for curr_pair in net_params:
        net_param_dict[curr_pair[0]] = curr_pair[1]

    # Get expt parameters
    expt_params = config.items('ExptParams')
    expt_param_dict = {}
    for curr_pair in expt_params:
        expt_param_dict[curr_pair[0]] = curr_pair[1]

    return [file_param_dict,
            net_param_dict,
            expt_param_dict]

def make_sure_outdir_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return

def make_outdir(main_dir, expt_dir):

    if not os.path.isdir(main_dir):
       make_sure_outdir_exists(main_dir)

    done = False
    suffix = ''
    while not done:
        curr_output_dir = os.path.join(main_dir, expt_dir + suffix)
        if not os.path.isdir(curr_output_dir):
           make_sure_outdir_exists(curr_output_dir)
           done = True
        elif suffix == '':
            suffix = '_v1'
        else:
            version = int(suffix[2:]) + 1
            suffix = '_v' + str(version)
    return curr_output_dir


def run_expt(expt_file):

    # Read cfg file params
    [file_param_dict,
     net_param_dict,
     expt_param_dict] = get_expt_params(expt_file)
        
    expt_set_dir = file_param_dict['expt_set_dir']
    expt_dir = file_param_dict['expt_dir']
    outdir = make_outdir(expt_set_dir, expt_dir)
    shutil.copy(expt_file, os.path.join(outdir,
                                        os.path.basename(expt_file)))


    [encoding_param_dict,
     encoding_module_param_dict,
     metric_param_dict] = get_encoding_params(file_param_dict['encoding_cfg'])
    shutil.copy(file_param_dict['encoding_cfg'],
                os.path.join(outdir,
                             os.path.basename(file_param_dict['encoding_cfg'])))

    try:
        from git import Repo

        keras_repo = Repo(os.path.dirname(keras.__path__[0]))
        keras_branch_name = str(keras_repo.active_branch)
        keras_commit_num = str(keras_repo.head.commit)

        backend_name = keras.backend._BACKEND
        if backend_name == 'theano':
            temp = keras.backend.theano_backend.theano.__version__.split('-')
            backend_version = '-'.join([temp[0],temp[1][0:8]])
        else:
            backend_version = "Unknown for " + backend_name
        
        repo = Repo('.')
        branch_name = str(repo.active_branch)
        commit_num = str(repo.head.commit)
        if repo.is_dirty():
            print('Files with Uncommitted Changes:\n')
            default_args = ['--abbrev=40', '--full-index', '--raw']
            raw_str = repo.git.diff(*default_args)
            raw_list = raw_str.split('\n')
            changed_file_list = ["    " + xx.split('\t')[-1]
                                 for xx in raw_list]
            changed_files = '\n'.join(changed_file_list)
            print(changed_files + '\n')
            changes = repo.git.diff(branch_name)
            print("Changes:\n")
            print(changes + '\n')
            print("=======================================\n")
        else:
            changed_files = "None\n"
            changes = "None\n"

        with open(os.path.join(outdir, 'git_info.txt'), 'w') as f:
            f.write("Keras Branch Name: " + keras_branch_name + '\n')
            f.write("Keras Commit Num: " + keras_commit_num[0:8] + '\n')
            f.write("Latent Branch Name: " + branch_name + '\n')
            f.write("Latent Commit Num: " + commit_num[0:8] + '\n')
            f.write("Backend: " + backend_name + '\n')
            f.write("Backend Version: " + backend_version + '\n')
            f.write("Changed Files: \n" + changed_files + '\n')
            f.write("Changes: \n")
            f.write(changes + '\n')
    except ImportError:
        pass

    # Run Expt
    start_time = datetime.datetime.now()
    expt_dm = DataManager(net_param_dict['output_activation'],
                          file_param_dict,
                          encoding_param_dict,
                          encoding_module_param_dict)
    expt_net = Cifar_Net(expt_dm, outdir,
                         net_param_dict,
                         expt_param_dict,
                         metric_param_dict)
    expt_log.stop_log()
    expt_net.train()
    stop_time = datetime.datetime.now()

    # Show run time (by wall clock)
    run_time = datetime.timedelta
    seconds = int(round(run_time.total_seconds(stop_time - start_time)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    start_str = "Start Time: %s" % (start_time.strftime("%H:%M:%S %p %A %Y-%m-%d"))
    stop_str = "Stop Time : %s" % (stop_time.strftime("%H:%M:%S %p %A %Y-%m-%d"))
    tot_str = "Run Time  : {:d}:{:02d}:{:02d}".format(hours, minutes, seconds)
    timing_info = '\n'.join([start_str, stop_str, tot_str])
    print (timing_info)
    expt_log.write(timing_info)
    #print("Start Time: %s" % (start_time.strftime("%H:%M:%S %p %A %Y-%m-%d")))
    #print("Stop Time : %s" % (stop_time.strftime("%H:%M:%S %p %A %Y-%m-%d")))
    #print("Run Time  : {:d}:{:02d}:{:02d}".format(hours, minutes, seconds))
    expt_log.close_log(outdir)

    return [expt_dm, expt_net]


if __name__ == '__main__':

    # Get cmd line args
    args = get_cmd_line_args()
    expt_file_name = args.config_file
    [dm, net] = run_expt(expt_file_name)
