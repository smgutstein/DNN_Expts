from __future__ import print_function
import argparse
import ConfigParser
import datetime
import errno
from expt_logger import Logger
import os
import shutil
import socket
import sys


# Capture output with theano/keras & gpu info
expt_log = Logger()

def is_number(in_str):
    try:
        float(in_str)
        return True
    except ValueError:
        return False

class Runner(object):

    def __init__(self):
        self.config = ConfigParser.ConfigParser()
        self.cmd_line_args = self.get_cmd_line_args()
        self.host_machine = socket.gethostname()

    def set_params(self):
        try:
            self.expt_file_name = self.cmd_line_args.next()
        except StopIteration as exception:
            return False
        
        self.config.read(self.expt_file_name)
        
        self.file_param_dict = self.get_param_dict('ExptFiles')        
        self.net_param_dict = self.get_param_dict('NetParams')
        self.expt_param_dict = self.get_param_dict('ExptParams')
        self.saved_param_dict = self.get_param_dict('SavedParams')
        
        self.expt_set_dir = self.file_param_dict['expt_set_dir']
        self.expt_dir = self.host_machine + "_" + self.file_param_dict['expt_dir']
        if not hasattr(self, 'outdir'):
            self.outdir = self.make_outdir(self.expt_set_dir, self.expt_dir)
        shutil.copy(self.expt_file_name,
                    os.path.join(self.outdir,
                                 os.path.basename(self.expt_file_name)))
        self.store_git_meta_data()

        self.config.read(self.file_param_dict['encoding_cfg'])
        self.encoding_param_dict = self.get_param_dict('Encoding')
        self.encoding_module_param_dict = self.get_param_dict('EncodingModuleParams')
        self.metric_param_dict = self.get_param_dict('MetricParams')

        if len(self.saved_param_dict) > 0:
            shutil.copy(self.file_param_dict['encoding_cfg'],    
                        os.path.join(self.outdir,
                                     os.path.basename(self.file_param_dict['encoding_cfg'])))            

        self.config.read(self.net_param_dict['optimizer_cfg'])
        temp = self.get_param_dict('OptimizerParams')
        self.optimizer_param_dict = {x:float(temp[x])
                                     if is_number(temp[x]) else temp[x]
                                     for x in temp}

        for x in self.optimizer_param_dict:
            if self.optimizer_param_dict[x] == 'None':
                self.optimizer_param_dict[x] = None
            elif self.optimizer_param_dict[x] == 'True':
                self.optimizer_param_dict[x] = True
            elif self.optimizer_param_dict[x] == 'False':
                self.optimizer_param_dict[x] = False

        shutil.copy(self.net_param_dict['optimizer_cfg'],
                    os.path.join(self.outdir,
                                 os.path.basename(self.net_param_dict['optimizer_cfg'])))

        self.expt_dm = DataManager(self.net_param_dict['output_activation'],
                                   self.file_param_dict,
                                   self.encoding_param_dict,
                                   self.encoding_module_param_dict,
                                   self.saved_param_dict,
                                   self.expt_param_dict)

        self.expt_net = NetManager(self.expt_dm, self.outdir,
                                   self.net_param_dict,
                                   self.expt_param_dict,
                                   self.metric_param_dict,
                                   self.optimizer_param_dict,
                                   self.saved_param_dict)
        return True

    


        
    def get_cmd_line_args(self):

        # Get config file from cmd line
        parser = argparse.ArgumentParser(
            description="Run Keras Expt With Specified Output Encoding")
        parser.add_argument('config_files', action='store',
                            type=str, nargs='*', default='', help="Specify cfg files for expt")
        parser.add_argument('--gpu', '-g', type=str, default='*',
                            action='store', help='chosen GPU')
        parser.add_argument('--dbg', action='store_true', help="Run Tensorflow CLI Debugger")

        cmd_line_args = parser.parse_args()

        # Choose specific GPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        if cmd_line_args.gpu != '*':
            os.environ['CUDA_VISIBLE_DEVICES'] = cmd_line_args.gpu
        os.environ['THEANO_FLAGS'] = 'floatX=32, mode=FAST_RUN, device=cuda'
        self.gpu = cmd_line_args.gpu

        # These imports are here, so that GPU is not selected before cmd line
        # arguments get parsed
        global keras
        global NetManager
        global DataManager

        import keras
        from net_manager import NetManager
        from data_manager import DataManager

        # Enable tensorflow debugger
        if cmd_line_args.dbg == True:
            print ("\n==========================\n")
            print ("DEBUG ENABLED")
            print ("\n==========================\n")
            import tensorflow as tf
            from tensorflow.python import debug as tf_debug
            import keras.backend as K
            K.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))


        for curr_config_file in (cmd_line_args.config_files):

            # Get input args from config file
            if not os.path.isfile(cmd_line_args.config_files[0]):
                print("Can't find %s. Is it a file?" % cmd_line_args.config_files)
                os._exit(1)

            yield curr_config_file


    def get_param_dict(self, dict_name):

        param_dict = {}
        try:
            params = self.config.items(dict_name)
            for curr_pair in params:
                param_dict[curr_pair[0]] = curr_pair[1]
        except ConfigParser.NoSectionError: 
            pass
        return param_dict

    def make_sure_outdir_exists(self, path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        return

    def make_outdir(self, main_dir, expt_dir):

        if not os.path.isdir(main_dir):
            self.make_sure_outdir_exists(main_dir)

        done = False
        suffix = ''
        while not done:
            curr_output_dir = os.path.join(main_dir, expt_dir + suffix + '_' + self.gpu)
            if not os.path.isdir(curr_output_dir):
                self.make_sure_outdir_exists(curr_output_dir)
                done = True
            elif suffix == '':
                suffix = '_v1'
            else:
                version = int(suffix[2:]) + 1
                suffix = '_v' + str(version)
        print ("Saving results to %s" % curr_output_dir)
        return curr_output_dir

    def store_git_meta_data(self):

        try:
            from git import Repo, InvalidGitRepositoryError

            try:
                keras_repo = Repo(os.path.dirname(keras.__path__[0]))
                keras_branch_name = str(keras_repo.active_branch)
                keras_commit_num = str(keras_repo.head.commit)
            except InvalidGitRepositoryError:
                print("No git repository found for keras")
                keras_branch_name = "Version: " + str(keras.__version__)
                keras_commit_num = "N/A"

            backend_name = keras.backend._BACKEND
            if backend_name == 'theano':
                temp = keras.backend.theano_backend.theano.__version__.split('-')
                if len(temp) > 1:
                    backend_version = '-'.join([temp[0], temp[1][0:8]])
                else:
                    backend_version = temp[0]
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
                #print("Changes:\n")
                #print(changes + '\n')
                print("=======================================\n")
            else:
                changed_files = "None\n"
                changes = "None\n"

            with open(os.path.join(self.outdir, 'git_info.txt'), 'w') as f:
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

        return
    

    def run_expt(self):

        # Run Expt
        start_time = datetime.datetime.now()
        expt_log.stop_log()
        expt_log.switch_log_file(self.outdir)
        self.expt_net.train()
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
        #expt_log.close_log(self.outdir)

        return 


if __name__ == '__main__':

    x = Runner()
    while x.set_params():
        x.run_expt()
    print ("Closing log " + x.outdir)
    expt_log.close_log(x.outdir)
