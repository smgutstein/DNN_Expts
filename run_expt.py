from __future__ import print_function
import argparse
import configparser
import ctypes
import datetime
import errno
from expt_logger import Logger
from operator import itemgetter
import os
import shutil
import socket
import sys

# Hmmm....think this might not be needed
try:
    from io import StringIO
except:
    from io import StringIO

# Hmmm....think this might not be needed
libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')


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
        self.config = configparser.ConfigParser()
        self.cmd_line_args = self.get_cmd_line_args()
        self.host_machine = socket.gethostname()

    def set_params(self):
        try:
            if sys.version_info < (3,0):
                self.expt_file_name = self.cmd_line_args.next()
            else:
                self.expt_file_name = self.cmd_line_args.__next__()
        except StopIteration as exception:
            return False
        
        self.config.read(self.expt_file_name)
        
        self.file_param_dict = self.get_param_dict('ExptFiles')        
        self.net_param_dict = self.get_param_dict('NetParams')
        self.expt_param_dict = self.get_param_dict('ExptParams')
        self.saved_param_dict = self.get_param_dict('SavedParams')
        self.trgt_task_param_dict = self.get_param_dict('TrgtTaskParams')
        self.preprocess_param_dict = self.get_param_dict('DataPreProcess')
        self.augment_param_dict = self.get_param_dict('DataAugment')
        self.regularizer_param_dict = self.get_param_dict('RegularizerParams')

        # ConfigParser doesn't allow for subsections, so this just seemed an
        # easier, though kludgy, way to add a new param dict to be used
        # in net_manager.py
        if len(self.regularizer_param_dict) != 0:
            self.net_param_dict['RegularizerParams'] = self.regularizer_param_dict
        else:
            self.net_param_dict['RegularizerParams'] = None

        # To allow quick testing without needing to change specified
        # epochs in cfg files
        if self.override_epochs > 0:
            self.expt_param_dict['epochs'] = self.override_epochs

        # Create output directory
        self.expt_set_dir = os.path.join(self.file_param_dict['root_expt_dir'],
                                         self.file_param_dict['expt_dir'],
                                         self.file_param_dict['expt_subdir'])

        # Possibly a bad idea, but I'm going to have output dir for tfer expts
        # automatically record which src task iteration was used, instead of
        # assigning that feature of the sub-dir name i a cfg file
        if 'saved_iter' in self.saved_param_dict:
             net_dir = os.path.join(self.saved_param_dict['saved_set_dir'],
                                    self.saved_param_dict['saved_dir'])

             
             if self.saved_param_dict['saved_iter'] == 'best':
                 self.expt_set_dir = os.path.join(self.expt_set_dir,'best')
                 best_file = [x for x in os.listdir(net_dir) if 'best' in x][0]
                 best_iter = best_file.split('_')[-1].split('.')[0]
                 self.expt_set_dir =  '_'.join([self.expt_set_dir,
                                                best_iter])
             else:
                 self.expt_set_dir = os.path.join(self.expt_set_dir,
                                              str(self.saved_param_dict['saved_iter']))

        
        self.expt_dir = self.host_machine + "_" + self.file_param_dict['expt_dir']
        if not hasattr(self, 'outdir'):
            self.outdir = self.make_outdir(self.expt_set_dir, self.expt_dir)
            
        # Make outdir for config/metadata
        self.metadata_dir = os.path.join(self.outdir, 'metadata')
        self.make_sure_outdir_exists(self.metadata_dir)
        checkpoint_dir = os.path.join(self.outdir, 'checkpoints')
        self.make_sure_outdir_exists(checkpoint_dir) # Used by net_manager
            
        # Copy files with metadata (cfg files) to output dir
        shutil.copy(self.expt_file_name,
                    os.path.join(self.metadata_dir,
                                 os.path.basename(self.expt_file_name)))
        if 'encoding_cfg' in self.file_param_dict:
            shutil.copy(self.file_param_dict['encoding_cfg'],    
                        os.path.join(self.metadata_dir, 'encoding.cfg'))
            
        if 'encoding_cfg' in self.trgt_task_param_dict:
            shutil.copy(self.file_param_dict['encoding_cfg'],    
                        os.path.join(self.metadata_dir, 'encoding.cfg'))

            self.config.read(self.trgt_task_param_dict['encoding_cfg'])
            self.trgt_task_param_dict['_EncodingParamDict'] = self.get_param_dict('Encoding')
            self.trgt_task_param_dict['_EncodingModuleParamDict'] = self.get_param_dict('EncodingModuleParams')
            self.trgt_task_param_dict['_MetricParams'] = self.get_param_dict('MetricParams')

            if 'encoding_cfg' in self.file_param_dict:
                shutil.copy(self.file_param_dict['encoding_cfg'],    
                            os.path.join(self.metadata_dir, 'orig_encoding.cfg'))
            
        self.store_git_meta_data()

        self.config.read(self.file_param_dict['encoding_cfg'])
        self.encoding_param_dict = self.get_param_dict('Encoding')
        self.encoding_module_param_dict = self.get_param_dict('EncodingModuleParams')
        self.metric_param_dict = self.get_param_dict('MetricParams')

        # Get optimizer params
        self.config.read(self.net_param_dict['optimizer_cfg'])
        self.optimizer_param_dict = self.get_param_dict('OptimizerParams')

        # Read/Create lr-schedule 
        if "lr_schedule" in self.optimizer_param_dict:
            # Read lr schedule and convert to list of 2-tuples (epoch, lr)
            lr_pairs =  self.optimizer_param_dict['lr_schedule'].split(")")
            lr_pairs = [x.strip(" ,()") for x in lr_pairs if len(x)>0]
            lr_schedule = [(int(y.split(",")[0]),
                            float(y.split(",")[1])) for y in lr_pairs]

            # Sort list and ensure initial lr is defined
            lr_schedule.sort(key=itemgetter(0))
            if lr_schedule[0][0] != 0:
               lr_schedule[0][0] = 0 # Ensures setting of initial lr
                
            self.optimizer_param_dict["lr_schedule"] = lr_schedule
            self.optimizer_param_dict["lr"] = lr_schedule[0][1]


        shutil.copy(self.net_param_dict['optimizer_cfg'],
                    os.path.join(self.metadata_dir,
                                 os.path.basename(self.net_param_dict['optimizer_cfg'])))

        self.expt_dm = DataManager(self.net_param_dict['output_activation'],
                                   self.file_param_dict,
                                   self.encoding_param_dict,
                                   self.encoding_module_param_dict,
                                   self.saved_param_dict,
                                   self.expt_param_dict,
                                   self.trgt_task_param_dict)

        self.expt_net = NetManager(self.expt_dm, self.outdir,
                                   self.metadata_dir,
                                   self.net_param_dict,
                                   self.expt_param_dict,
                                   self.metric_param_dict,
                                   self.optimizer_param_dict,
                                   self.saved_param_dict,
                                   self.trgt_task_param_dict,
                                   self.preprocess_param_dict,
                                   self.augment_param_dict,
                                   self.nocheckpoint)
        return True

    


        
    def get_cmd_line_args(self):

        # Get config file from cmd line
        parser = argparse.ArgumentParser(
            description="Run Keras Expt With Specified Output Encoding")
        parser.add_argument('config_files', action='store',
                            type=str, nargs='*', default='', help="Specify cfg files for expt")
        parser.add_argument('--gpu', '-g', type=str, default='',
                            action='store', help='specify GPU')
        parser.add_argument('--dbg', action='store_true', help="Run Tensorflow CLI Debugger")
        parser.add_argument('--epochs', '-e', type=int, default=0,
                            action='store', help='override epochs from cfg file')
        parser.add_argument('--silent', '-s', action='store_true',
                            help="reduce diagnostic output - e.g. gitinfo and info printed to screen")
        parser.add_argument('--nocheckpoint', '-n', action='store_true',
                            help="don't save trained nets")

        cmd_line_args = parser.parse_args()
        self.nocheckpoint = cmd_line_args.nocheckpoint

        # Turn off printing if requested (i.e. for batch jobs)
        if cmd_line_args.silent:
            sys.stdout = open(os.devnull,'w')
            

        print ("CMD LINE ARGS:")
        temp = vars(cmd_line_args)
        for temp_arg in temp:
            print(temp_arg,":",temp[temp_arg])


        # Choose specific GPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        if cmd_line_args.gpu != '':
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

        # Check if number of desired epochs different than in cfg file
        self.override_epochs = cmd_line_args.epochs

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

            # Convert numeric strings to float
            param_dict = {x:float(param_dict[x])
                               if is_number(param_dict[x])
                                 else param_dict[x]
                               for x in param_dict}

            # Convert non-numeric strings to correct variable types
            for x in param_dict:
                if param_dict[x] == 'None':
                    param_dict[x] = None
                elif param_dict[x] == 'True':
                    param_dict[x] = True
                elif param_dict[x] == 'False':
                    param_dict[x] = False

        except configparser.NoSectionError:
            # Return empty dict if section missing
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
        suffix = '_v0'
        while not done:
            curr_output_dir = os.path.join(main_dir, expt_dir + suffix) # + '_' + self.gpu)
            if not os.path.isdir(curr_output_dir):
                self.make_sure_outdir_exists(curr_output_dir)
                done = True
            #elif suffix == '':
            #    suffix = '_v1'
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

                print("=======================================\n")
            else:
                changed_files = "None\n"
                changes = "None\n"

            with open(os.path.join(self.metadata_dir, 'git_info.txt'), 'w') as f:
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
        expt_log.switch_log_file(self.metadata_dir)
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
        expt_log.write(timing_info + '\n')
        score_str = "{:5.2f}%".format(self.expt_net.best_score*100).strip()
        epoch_str = "{:4d}".format(self.expt_net.best_epoch).strip()
        result_str = "Peak Accuracy: " + score_str + " at epoch " + epoch_str + '\n'
        print(result_str)
        expt_log.write(result_str)
        #expt_log.close_log(self.outdir)

        return 


if __name__ == '__main__':

    x = Runner()
    while x.set_params():
        x.run_expt()
    print ("Closing log " + x.metadata_dir)
    expt_log.close_log(x.metadata_dir)
    sys.stdout = sys.__stdout__
