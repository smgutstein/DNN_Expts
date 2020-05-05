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

        self.notes_dict = dict()
        
        # Get Expt Params
        self.config.read(self.expt_file_name)
        self.file_param_dict = self.get_param_dict('ExptFiles')        
        self.net_param_dict = self.get_param_dict('NetParams')
        self.expt_param_dict = self.get_param_dict('ExptParams')
        self.saved_param_dict = self.get_param_dict('SavedParams')
        self.trgt_task_param_dict = self.get_param_dict('TrgtTaskParams')
        self.preprocess_param_dict = self.get_param_dict('DataPreprocessParams')
        self.augment_param_dict = self.get_param_dict('DataAugmentParams')

        # Get notes from expt cfg file
        temp_dict = self.get_param_dict('Notes')
        if 'notes' in temp_dict:
            self.notes_dict["Expt Notes"] = temp_dict['notes']
            self.config.remove_section('Notes')
        
        # To allow quick testing without needing to change specified
        # epochs in cfg files
        if self.override_epochs > 0:
            self.expt_param_dict['epochs'] = self.override_epochs
            self.config.remove_section('Notes')

        # Get Encoding Params
        self.config.read(self.file_param_dict['encoding_cfg'])
        self.encoding_param_dict = self.get_param_dict('Encoding')
        self.encoding_module_param_dict = self.get_param_dict('EncodingModuleParams')
        self.metric_param_dict = self.get_param_dict('MetricParams')

        # Get notes from encoding cfg file
        temp_dict = self.get_param_dict('Notes')
        if 'notes' in temp_dict:
            self.notes_dict["Encoding Notes"] = temp_dict['notes']
            self.config.remove_section('Notes')

        # Get Optimizer Params
        self.config.read(self.net_param_dict['optimizer_cfg'])
        self.optimizer_param_dict = self.get_param_dict('OptimizerParams')
        self.lrschedule_param_dict = self.get_param_dict('LRSchedParams')
        self.regularizer_param_dict = self.get_param_dict('RegularizerParams')

        # Get notes from optimizer cfg file
        temp_dict = self.get_param_dict('Notes')
        if 'notes' in temp_dict:
            self.notes_dict["Optimizer Notes"] = temp_dict['notes']

        # Record Changes From Data Augment/Preprocess Args
        temp_dict = dict()
        #  Get Augmentation Changes
        for curr_key in self.augment_param_dict:
            if self.augment_param_dict[curr_key] != ImageDataGen_args[curr_key]:
                temp_dict[curr_key] = self.augment_param_dict[curr_key]

        #  Get Preprocessing Changes
        for curr_key in self.preprocess_param_dict:
            if self.preprocess_param_dict[curr_key] != ImageDataGen_args[curr_key]:
                temp_dict[curr_key] = self.preprocess_param_dict[curr_key]

        #  Create Output String Recording Changes
        self.data_mod_str = ""
        for x in temp_dict:
            self.data_mod_str += "  " + x + ": " + str(temp_dict[x]) + "\n"
        if len(self.data_mod_str) == 0:
            self.data_mod_str = "\nData Modifications: None\n"
        else:
            self.data_mod_str = "\nData Modifications:\n" + self.data_mod_str

        # Create Output String for ad hoc notes
        self.notes_str = ""
        for x in sorted(self.notes_dict):
            self.notes_str += "  " + x + ": " + str(self.notes_dict[x]) + "\n"
        if len(self.notes_str) == 0:
            self.notes_str = "Notes: None\n"
        else:
            self.notes_str = "Notes:\n" + self.notes_str

        # Create output directory
        self.expt_set_dir = os.path.join(self.file_param_dict['root_expt_dir'],
                                         self.file_param_dict['expt_dir'],
                                         self.file_param_dict['expt_subdir'])

        # Possibly a bad idea, but I'm going to have output dir for tfer expts
        # automatically record which src task iteration was used, instead of
        # assigning that feature of the sub-dir name in a cfg file
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

        # Make output_dir, if necessary        
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
            
        if 'optimizer_cfg' in self.file_param_dict:
            shutil.copy(self.file_param_dict['optimizer_cfg'],    
                        os.path.join(self.metadata_dir, 'optimizer.cfg'))
            
        if 'encoding_cfg' in self.trgt_task_param_dict:
            shutil.copy(self.file_param_dict['encoding_cfg'],    
                        os.path.join(self.metadata_dir, 'encoding.cfg'))

            # Need to overwrite source task encoding
            self.config.read(self.trgt_task_param_dict['encoding_cfg'])
            self.trgt_task_param_dict['_EncodingParamDict'] = self.get_param_dict('Encoding')
            self.trgt_task_param_dict['_EncodingModuleParamDict'] = self.get_param_dict('EncodingModuleParams')
            self.trgt_task_param_dict['_MetricParams'] = self.get_param_dict('MetricParams')

            if 'encoding_cfg' in self.file_param_dict:
                shutil.copy(self.file_param_dict['encoding_cfg'],    
                            os.path.join(self.metadata_dir, 'orig_encoding.cfg'))

        shutil.copy(self.net_param_dict['optimizer_cfg'],
                    os.path.join(self.metadata_dir,
                                 os.path.basename(self.net_param_dict['optimizer_cfg'])))

        self.expt_dm = DataManager(self.net_param_dict['output_activation'],
                                   self.file_param_dict,
                                   self.encoding_param_dict,
                                   self.encoding_module_param_dict,
                                   self.saved_param_dict,
                                   self.expt_param_dict,
                                   self.trgt_task_param_dict,
                                   self.preprocess_param_dict,
                                   self.augment_param_dict)

        self.expt_net = NetManager(self.expt_dm, self.outdir,
                                   self.metadata_dir,
                                   self.net_param_dict,
                                   self.expt_param_dict,
                                   self.metric_param_dict,
                                   self.optimizer_param_dict,
                                   self.lrschedule_param_dict,
                                   self.regularizer_param_dict,
                                   self.saved_param_dict,
                                   self.trgt_task_param_dict,
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
        global ImageDataGen_args

        import keras
        from net_manager import NetManager
        from data_manager import DataManager
        from data_manager import ImageDataGen_args

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
                if str(param_dict[x]).lower() == 'none':
                    param_dict[x] = None
                elif str(param_dict[x]).lower() == 'true':
                    param_dict[x] = True
                elif str(param_dict[x]).lower() == 'false':
                    param_dict[x] = False

            # Convert strings representing lists of floats to floats
            # TBD: Generalize this to more than floats. Perhaps turn
            # above 2 conversion loops to separate method
            for x in param_dict:
                if isinstance(param_dict[x], str):
                    temp = param_dict[x].strip()
                    if (len(temp) > 0 and
                        ((temp[0] == '[' and temp[-1] == ']') or
                         (temp[0] == '(' and temp[-1] == ')'))):
                        temp = temp[1:-1]
                        temp = temp.split(',')
                        temp = [float(y) for y in temp]
                        param_dict[x] = temp

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

            try:
                backend_name = keras.backend.backend()
            except:
                backend_name = keras.backend._BACKEND
            if backend_name == 'theano':
                temp = keras.backend.theano_backend.theano.__version__.split('-')
                if len(temp) > 1:
                    backend_version = '-'.join([temp[0], temp[1][0:8]])
                else:
                    backend_version = temp[0]
            elif backend_name == 'tensorflow':
                backend_version = keras.backend.tensorflow_backend.tf.__version__
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

    def store_env_data(self):

        major, minor, micro, rlease, cereal = sys.version_info
        vers = '.'.join([str(x) for x in [major, minor, micro]]) 
        out_str = "Python: " + vers + "  Release: " + rlease + "  Serial: " + str(cereal) + "\n" 
        indent_str = "    "
        
        if os.path.exists(os.path.join(sys.prefix, 'conda-meta')):
            out_str += "Found Anaconda Environment:\n"
            env_name = os.path.basename(sys.prefix)
            out_str += indent_str + "Name: " + env_name + '\n'            
        else:
            out_str += "Did Not Find Anaconda Environment:\n"

        temp_list = sys.path
        for temp in temp_list:
            if os.path.basename(temp) == 'site-packages':
                site_pack_dir = temp
                break
        out_str += indent_str + "Installed Packages:\n"
        installed_packages = os.listdir(site_pack_dir)
        for curr_pack in sorted(installed_packages):
            out_str += 2*indent_str + curr_pack + '\n'

        with open(os.path.join(self.metadata_dir, 'env_info.txt'), 'w') as f:
            f.write(out_str)

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
        print(self.data_mod_str)
        print(self.notes_str)
        print(result_str)
        expt_log.write(self.data_mod_str+'\n')
        expt_log.write(self.notes_str+'\n')
        expt_log.write(result_str)
        #expt_log.close_log(self.outdir)
        self.store_git_meta_data()
        self.store_env_data()

        return 

if __name__ == '__main__':

    x = Runner()
    while x.set_params():
        x.run_expt()
    print ("Closing log " + x.metadata_dir)
    expt_log.close_log(x.metadata_dir)
    sys.stdout = sys.__stdout__
