from __future__ import print_function

import curses
import errno
import importlib
import inspect
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import model_from_json, model_from_yaml
from keras_loggers import TrainingMonitor, ModelCheckpoint
from operator import itemgetter
import os
import pickle
import shutil
import sys

def is_int(in_str):
    try:
        int(in_str)
        return True
    except ValueError:
        return False

class NetManager(object):

    def __init__(self, data_manager,
                 expt_dir,
                 net_param_dict,
                 expt_param_dict,
                 metric_param_dict,
                 optimizer_param_dict,
                 saved_param_dict,
                 data_augmentation=True,
                 save_iters=True,
                 save_best_n=5):

        self.init_epoch = 0
        self.epochs = int(expt_param_dict['epochs'])
        self.data_manager = data_manager
        self.expt_dir = expt_dir
        self.expt_prefix = os.path.basename(expt_dir)
        self.data_augmentation = data_augmentation
        self.save_iters = save_iters
        if self.save_iters:
            self.epochs_per_recording = int(expt_param_dict['epochs_per_recording'])
        else:
            self.epochs_per_recording = self.epochs
            
        self.tot_rec_sets = self.epochs/self.epochs_per_recording
        self.save_best_n = save_best_n
        self.best_n = 0

        # Ensure expt output dir exists
        if expt_dir is not None:
            try:
                os.makedirs(expt_dir)
            except OSError:
                if not os.path.isdir(expt_dir):
                    raise
                
        # Make optimizer
        optimizer_module = optimizer_param_dict.pop('optimizer_module')
        optimizer = optimizer_param_dict.pop('optimizer')
        temp = importlib.import_module(optimizer_module)
        optimizer_fnc = getattr(temp, optimizer)
        self.opt = optimizer_fnc(optimizer_param_dict)

        # Get Loss Function
        if 'loss_fnc' in net_param_dict:
            self.loss_fnc = net_param_dict['loss_fnc']
        else:
            self.loss_fnc = 'mean_squared_error'

        # Prepare standard training
        print("Standard training")
        self.nb_output_nodes = data_manager.nb_code_bits
        print("Initializing data manager ...")
        self.data_manager = data_manager

        # Import accuracy function
        temp = importlib.import_module(metric_param_dict['metrics_module'])
        metric_fnc = getattr(temp, metric_param_dict['accuracy_metric'])
        metric_fnc_args = inspect.getargspec(metric_fnc)
        if metric_fnc_args.args == ['y_encode']:
            metric_fnc = metric_fnc(self.data_manager.encoding_matrix)
        self.acc_metric = metric_param_dict['accuracy_metric']
        metric_fnc.__name__ = 'acc'  # Hacky solution to make sure
                                     # keras output strings are unaffected
                                     # by use of local_metrics

        print("Initializing architecture ...")
        self.model = self.init_model_architecture(net_param_dict,
                                                  saved_param_dict)
        self.check_for_saved_model_weights(net_param_dict,
                                           saved_param_dict)

        # Save net architecture
        json_str = self.model.to_json()
        model_file = os.path.join(self.expt_dir,
                                  self.expt_prefix + "_init.json")
        open(model_file, "w").write(json_str)

        # Compile model
        print("Compiling model ...")
        self.model.compile(loss=self.loss_fnc, #'mean_squared_error',
                           optimizer=self.opt,
                           metrics=[metric_fnc])

        try:
            from keras.utils import plot_model
            # Write the network architecture visualization graph to disk
            model_img_file = os.path.join(self.expt_dir,
                                      self.expt_prefix + "_image.png")
            plot_model(self.model, to_file=model_img_file, show_shapes=True)
            print ("Saved image of architecture to", model_img_file)
        except ImportError, e:
            # Prob'ly need to install pydot
            print (e.message)
            print ("Not saving graphical image of net")
        except OSError, e:
            # Prob'ly need Graphviz
            print (e.message)
            print ("Not saving graphical image of net")


        # Summarize            
        self.summary()
        print (self.data_manager.get_targets_str_sign())

    def summary(self):
        print ("\n============================================================\n")
        print ("Expt Info:\n")
        print ("NB Epochs:", self.epochs)
        print ("Expt Dir:", self.expt_dir)
        print ("Expt Prefix:", self.expt_prefix)

        print ("\nModel:")
        self.model.summary()
        print ("\n============================================================\n")

    def init_model_architecture(self, net_param_dict, saved_param_dict):
        # Import Architecture
        if ('arch_module' in net_param_dict and
            len(net_param_dict['arch_module']) > 0):

            # Create net architecure from general module
            if K.image_data_format() != 'channels_last':
                input_shape = (self.data_manager.img_channels,
                               self.data_manager.img_rows,
                               self.data_manager.img_cols)
            else:
                input_shape = (self.data_manager.img_rows,
                               self.data_manager.img_cols,
                               self.data_manager.img_channels)

            mod_name = net_param_dict['arch_module']
            if mod_name[0] != '.':
                mod_name = '.' + mod_name
            temp = importlib.import_module(mod_name, 'net_architectures')
            build_architecture = getattr(temp, "build_architecture")

            try:
                 arch = build_architecture(input_shape,
                                           self.nb_output_nodes,
                                           net_param_dict['output_activation'])
            except curses.error,e:
                print('\nError:')
                print (e.message)
                print ("Check to ensure you're using a POSIX enabled terminal - i.e. Works with POSIX termios calls")
                print ('\n\n')
                sys.exit()
                
            
            return arch

        elif (len(saved_param_dict) > 0 and
              'saved_arch_format' in saved_param_dict and
              len(saved_param_dict['saved_arch_format']) > 0):

            # Load architecture
            with open(os.path.join(saved_param_dict['saved_set_dir'],
                                   saved_param_dict['saved_dir'],
                                   saved_param_dict['saved_dir'] + '.' +
                                   saved_param_dict['saved_arch']), 'r') as f:

                if saved_param_dict['saved_arch_format'][-4:] == 'json':
                    json_str = f.read()
                    return model_from_json(json_str)
                elif saved_param_dict['saved_arch'][-4:] == 'yaml':
                    yaml_str = f.read()
                    return model_from_yaml(yaml_str)
                else:
                    # Error
                    print("No architecure was specified in config file, either by 'arch_module' or 'saved_arch'")
                    sys.exit(0)

    def check_for_saved_model_weights(self, net_param_dict, saved_param_dict):

        # Load weights (if necessary)
        if (len(saved_param_dict) == 0 or
            ('saved_set_dir' not in saved_param_dict or
             'saved_dir' not in saved_param_dict or
             'saved_iter' not in saved_param_dict
             )):

            # No saved weights - training from scratch
            self.init_epoch = 0
            print("Starting with weights from epoch %d" % self.init_epoch)
            wt_file = None

        else:
            net_dir = os.path.join(saved_param_dict['saved_set_dir'],
                                   saved_param_dict['saved_dir'])
            net_iter = saved_param_dict['saved_iter']

            if ('saved_weights_file' not in saved_param_dict):

                if net_iter.lower().strip() == 'last':

                    # Load from final iteration
                    wt_files = [(os.path.join(net_dir, x),
                                 os.stat(os.path.join(net_dir, x)).st_mtime)
                                 for x in os.listdir(net_dir)
                                 if x[-3:] == '.h5']
                    wt_files = sorted(wt_files, key=itemgetter(1))
                    wt_file = wt_files[-1][0]
                    self.init_epoch = int(wt_file.split('_')[-1].split('.')[0])

                elif is_int(net_iter):
                    # Load weights from specified iteration (or closest
                    # iteration prior to specified iteration)
                    self.init_epoch = int(net_iter)
                    wt_file = os.path.join(net_dir, saved_param_dict['saved_dir'] +
                                           '_weights_' + saved_param_dict['saved_iter'] +
                                           '.h5')

            elif 'saved_weights_file' in saved_param_dict:
                # Load weights from specific file name
                # Assumes that default method of naming weight files
                # was used so that starting epoch may be read off as
                # number after last '_', before extension 
                wt_file = os.path.join(net_dir, saved_param_dict['saved_weights'])

                if int(net_iter):
                    self.init_epoch = int(net_iter)
                else:
                    sys.exit("Iteration number must be given when " +
                             "a saved weight file is explicitly named")
            print("Starting with weights from epoch %d" % self.init_epoch)
            if wt_file is not None:
               print("   Loading wts from %s" % wt_file)
            self.model.load_weights(wt_file)

            # Hacky way of ensuring that continuing training from a saved point
            # does not result in a faux encoding change
            dm = self.data_manager
            dm.curr_encoding_info['encoding_dict'] = dm.encoding_dict
            dm.curr_encoding_info['label_dict'] = dm.label_dict
            dm.curr_encoding_info['meta_encoding_dict'] = dm.meta_encoding_dict

            # Copy info from orig expt to current expt. dir
            orig_expt_copy_dir = os.path.join(self.expt_dir, "seed_expt")
            try:
                os.makedirs(orig_expt_copy_dir)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

            orig_expt_files = os.listdir(net_dir)
            dup_files = [x for x in orig_expt_files
                         if 'h5' not in x or x == wt_file.split('/')[-1]]
            for curr_file in dup_files:
                shutil.copy2(os.path.join(net_dir, curr_file), orig_expt_copy_dir)

    def get_init_condits(self):
        dm = self.data_manager
        if not dm.train_data_generator:
            init_train_loss, init_train_acc = \
            self.model.evaluate(dm.X_train,
                                dm.Y_train)
            init_test_loss, init_test_acc = \
            self.model.evaluate(dm.X_test,
                                dm.Y_test)
        else:
            init_train_loss, init_train_acc = \
            self.model.evaluate_generator(dm.train_data_generator,
                                          steps=dm.train_data_generator.batches_per_epoch
                                          )
            init_test_loss, init_test_acc = \
            self.model.evaluate_generator(dm.test_data_generator,
                                          steps=dm.test_data_generator.batches_per_epoch
                                          )


        print("\nInit loss and acc:                             loss: ",
              "%0.5f - acc: %0.5f - val_loss: %0.5f - val_acc: %0.5f" %
              (init_train_loss, init_train_acc,
               init_test_loss, init_test_acc))

        return (init_train_loss, init_train_acc,
               init_test_loss, init_test_acc)

            
    def train(self, data_augmentation=True):

        results_path = os.path.join(self.expt_dir, 'results.txt')
        fig_path = [os.path.join(self.expt_dir, 'results_acc.png'),
                    os.path.join(self.expt_dir, 'results_loss.png')]
        json_path = os.path.join(self.expt_dir, 'results.json')
        checkpoint_path = os.path.join(self.expt_dir,"checkpoint")
        training_monitor = TrainingMonitor(fig_path, jsonPath=json_path,
                                           resultsPath = results_path)
        checkpointer = ModelCheckpoint(checkpoint_path, verbose=1,
                                       data_manager=self.data_manager,
                                       period=self.epochs_per_recording)
        callbacks = [training_monitor, checkpointer]

        (init_train_loss, init_train_acc,
               init_test_loss, init_test_acc) = self.get_init_condits()
        
        training_monitor.on_train_begin()
        training_monitor.on_epoch_end(epoch=0,logs = {'loss': init_train_loss,
                                                      'acc':  init_train_acc,
                                                      'val_loss': init_test_loss,
                                                      'val_acc': init_test_acc})

        # Train Model
        dm = self.data_manager
        print(dm.data_generator_info)
        if not dm.train_data_generator:
            results = self.model.fit(dm.X_train,
                                     dm.Y_train,
                                     batch_size=dm.batch_size,
                                     epochs=self.epochs,
                                     validation_data=(dm.X_test,
                                                      dm.Y_test),
                                     callbacks=callbacks,
                                     shuffle=True)
            
        else:
            # fit the model on the batches generated by datagen.flow()
            results = self.model.fit_generator(dm.train_data_generator,
                                               steps_per_epoch=dm.train_data_generator.batches_per_epoch,
                                               epochs=self.epochs,
                                               validation_data=dm.test_data_generator,
                                               validation_steps=dm.test_data_generator.batches_per_epoch,
                                               callbacks = callbacks,
                                               shuffle=True)
