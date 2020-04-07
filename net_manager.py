from __future__ import print_function

import curses
import errno
import importlib
import inspect
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import model_from_json, model_from_yaml
from keras_loggers import TrainingMonitor, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten 
from net_architectures.sgActivation import Activation

from lr_scheduler import StepLearningRateScheduler
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
                 metadata_dir,
                 net_param_dict,
                 expt_param_dict,
                 metric_param_dict,
                 optimizer_param_dict,
                 saved_param_dict,
                 trgt_task_param_dict,
                 preprocess_param_dict,
                 augment_param_dict,
                 nocheckpoint,
                 data_augmentation=True,
                 save_iters=True,
                 save_best_n=5):

        self.init_epoch = 0
        self.epochs = int(expt_param_dict['epochs'])
        self.data_manager = data_manager
        self.expt_dir = expt_dir
        self.metadata_dir = metadata_dir
        self.expt_prefix = os.path.basename(expt_dir)
        self.data_augmentation = data_augmentation
        self.preprocess_param_dict = preprocess_param_dict
        self.augment_param_dict = augment_param_dict
        self.nocheckpoint = nocheckpoint
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
        if "lr_schedule" in optimizer_param_dict:
            self.lr_schedule = optimizer_param_dict['lr_schedule']
            self.lr_schedule.sort(key=itemgetter(0))
            lr_init = self.lr_schedule[0][1]
            optimizer_param_dict['lr'] = lr_init
        else:
            self.lr_schedule = None

        optimizer_module = optimizer_param_dict.pop('optimizer_module')
        optimizer = optimizer_param_dict.pop('optimizer')
        temp = importlib.import_module(optimizer_module)
        optimizer_fnc = getattr(temp, optimizer)
        self.opt = optimizer_fnc(optimizer_param_dict)

        # Check if using optimzier designed to change learning rate
        # according to schedule - figure out why kwargs didn't work
        if 'set_batches_per_epoch' in dir(self.opt):
            self.opt.set_batches_per_epoch(self.data_manager.batches_per_epoch)

        # Get Loss Function
        if 'loss_fnc' in net_param_dict:
            self.loss_fnc = net_param_dict['loss_fnc']
        else:
            self.loss_fnc = 'mean_squared_error'

        # Get Loss Function Regularizer
        if net_param_dict['RegularizerParams'] != None:
          reg_param_dict = net_param_dict['RegularizerParams']
          reg_module = reg_param_dict['regularizer_module']
          reg_fnc_name = reg_param_dict['regularizer_fnc']
          reg_args = {k:v for k,v in reg_param_dict.items()
                      if k != "regularizer_module" and  k != "regularizer_fnc"}
          temp = importlib.import_module(reg_module)
          reg_fnc = getattr(temp, reg_fnc_name)
          net_param_dict['regularizer'] = reg_fnc(**reg_args)
        else:
          net_param_dict['regularizer'] = None  
          

        # Prepare standard training
        print("Standard training")
        self.nb_output_nodes = data_manager.nb_code_bits
        self.src_nb_output_nodes = data_manager.src_nb_code_bits
        print("Initializing data manager ...")
        self.data_manager = data_manager

        # Import accuracy function
        temp = importlib.import_module(metric_param_dict['metrics_module'])
        metric_fnc = getattr(temp, metric_param_dict['accuracy_metric'])
        metric_fnc_args = inspect.getargspec(metric_fnc)
        # NOTE: Need top ensure correct acc fnc is obtained whjen recovering
        # previous encoding
        if metric_fnc_args.args == ['y_encode']:
            metric_fnc = metric_fnc(self.data_manager.encoding_matrix)
            print ("Warning: Need to ensure correct acc", end= ' ')
            print ("function is obtained with reuse of encoding")
        self.acc_metric = metric_fnc.__name__
        self.train_acc_str = self.acc_metric
        self.val_acc_str = 'val_' + self.train_acc_str

        # Create net either from scratch, continuing with
        # a saved net, ot using a saved net for transfer
        print("Initializing architecture ...")
        self.net_arch_file = None
        self.model = self.init_model_architecture(net_param_dict,
                                                  saved_param_dict)
        self.check_for_saved_model_weights(net_param_dict,
                                           saved_param_dict)
        self.check_for_transfer(trgt_task_param_dict,
                                net_param_dict)

        # Save net architecture
        #  Copies module with code for net to metadata file
        arch_file_name = os.path.basename(self.net_arch_file)
        shutil.copy2(self.net_arch_file,
                     os.path.join(self.metadata_dir, arch_file_name))
        #  Copies net architecture to json file
        json_str = self.model.to_json()
        model_file = os.path.join(self.expt_dir, "checkpoints",
                                  "init_architecture.json")
        open(model_file, "w").write(json_str)

        # Compile model
        print("Compiling model ...")
        self.model.compile(loss=self.loss_fnc, #'mean_squared_error',
                           optimizer=self.opt,
                           metrics=[metric_fnc])

        try:
            from keras.utils import plot_model
            # Write the network architecture visualization graph to disk
            model_img_file = os.path.join(self.expt_dir,"metadata",
                                      self.expt_prefix + "_image.png")
            plot_model(self.model, to_file=model_img_file, show_shapes=True)
            print ("Saved image of architecture to", model_img_file)
        except ImportError as e:
            # Prob'ly need to install pydot
            print (e)
            print ("Not saving graphical image of net")
        except OSError as e:
            # Prob'ly need Graphviz
            print (e)
            print ("Not saving graphical image of net")


        # Summarize            
        self.summary()
        print (self.data_manager.get_targets_str_sign())
        print (self.data_manager.get_data_classes_summary_str())

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

            # Get arch file to save in experimental results dir
            self.net_arch_file = temp.__file__
            if self.net_arch_file[-4:] == '.pyc':
                self.net_arch_file = self.net_arch_file[:-1]

            try:
                 arch = build_architecture(input_shape,
                                           self.src_nb_output_nodes,
                                           net_param_dict['output_activation'],
                                           regularizer=net_param_dict['regularizer'])
            except curses.error as e:
                print('\nError:')
                print (e.message)
                print ("Check to ensure you're using a POSIX", end = ' ')
                print ("enabled terminal - i.e. Works with POSIX termios calls")
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
                    print("No architecure was specified in ", end=' ')
                    print ("config file, either by 'arch_module' or 'saved_arch'")
                    sys.exit(0)

                self.net_arch_file = os.path.join(saved_param_dict['saved_set_dir'],
                                                  saved_param_dict['saved_dir'],
                                                  saved_param_dict['saved_dir'] + '.' +
                                                  saved_param_dict['saved_arch'])

    def check_for_transfer(self, trgt_task_param_dict,
                           net_param_dict):
        if len(trgt_task_param_dict) == 0:
            return

        num_resets = int(trgt_task_param_dict['num_reset_layers'])
        pen_ult_nodes = trgt_task_param_dict['penultimate_node_list']
        pen_ult_nodes = pen_ult_nodes.split('[')[1].strip()
        pen_ult_nodes = [x.strip()
                         for x in pen_ult_nodes.split(']')[0].split(',')]
        pen_ult_nodes = [x for x in pen_ult_nodes if x.isdigit()]

        
        if 'output_activation' in trgt_task_param_dict:
            output_activation = trgt_task_param_dict['output_activation']
        else:
            output_activation = net_param_dict['output_activation']
        
        ctr = 0
        
        while ctr < num_resets:
            #Returns layer info, but keeps layer in net
            pop_layer = self.model.layers.pop() 
            print("Popping ",pop_layer.name," ..... ",end=' ')

            # Count layers *with* trainable weights that get popped
            if len(pop_layer.weights) > 0:
                ctr += 1
            print("Popped")
            
            #Removes layer from net, but returns no info 
            self.model.pop()

        # Add intermediate layers
        if len(pen_ult_nodes) > 0:
            for curr in pen_ult_nodes:
                self.model.add(Dense(int(curr)))
                self.model.add('Activation'('relu'))
                self.model.add(Dropout(0.5))
            else:
                pass

        # Add final classification layer
        self.model.add(Dense(self.nb_output_nodes))
        self.model.add(Activation(output_activation,
                                  name=output_activation + '_tfer_out'))
                
        temp=0


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

                if 'last' in str(net_iter).lower().strip():

                    # Load from final iteration
                    wt_files = [(os.path.join(net_dir, x),
                                 os.stat(os.path.join(net_dir, x)).st_mtime)
                                 for x in os.listdir(net_dir)
                                 if x[-3:] == '.h5']
                    wt_files = sorted(wt_files, key=itemgetter(1))
                    wt_file = wt_files[-1][0]
                    self.init_epoch = int(wt_file.split('_')[-1].split('.')[0])

                elif 'best' in str(net_iter).lower().strip():
                    best_file = [x for x in os.listdir(net_dir) if 'best' in x][0]
                    wt_file = os.path.join(net_dir, best_file)

                elif is_int(net_iter.split('_')[0]):
                    # Load weights from specified iteration (or closest
                    # iteration prior to specified iteration)
                    self.init_epoch = int(net_iter.split('_')[0])
                    #wt_file = os.path.join(net_dir, saved_param_dict['saved_dir'] +
                    wt_file = os.path.join(net_dir,
                                           'checkpoint_weights_' +
                                           str(self.init_epoch) +
                                           '.h5')

            elif 'saved_weights_file' in saved_param_dict:
                # Load weights from specific file name
                # Assumes that default method of naming weight files
                # was used so that starting epoch may be read off as
                # number after last '_', before extension 
                wt_file = os.path.join(net_dir, saved_param_dict['saved_weights_file'])

                if int(net_iter.split('_')[0]):
                    self.init_epoch = int(net_iter.split('_')[0])
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
                # If nocheckpoint is True, don't save copy of source net
                # (i.e. init condits for trgt net)
                if curr_file[-3:] != '.h5' or not self.nocheckpoint:
                    shutil.copy2(os.path.join(net_dir, curr_file), orig_expt_copy_dir)
                elif  curr_file[-3:] == '.h5' and self.nocheckpoint:
                    # Create soft link to original src file
                    curr_src = os.path.abspath(os.path.join(net_dir, curr_file))
                    curr_trgt = os.path.join(orig_expt_copy_dir, curr_file)
                    os.symlink(curr_src, curr_trgt)

    def get_init_condits(self, train_data_generator,
                         test_data_generator):
        dm = self.data_manager
        if not dm.data_generator_module:
            init_train_loss, init_train_acc = \
            self.model.evaluate(dm.X_train,
                                dm.Y_train)
            init_test_loss, init_test_acc = \
            self.model.evaluate(dm.X_test,
                                dm.Y_test)
        else:
            init_train_loss, init_train_acc = \
            self.model.evaluate_generator(train_data_generator,
                                          steps=train_data_generator.batches_per_epoch
                                          )
            init_test_loss, init_test_acc = \
            self.model.evaluate_generator(test_data_generator,
                                          steps=test_data_generator.batches_per_epoch
                                          )

        print("\nInit loss and acc:                             loss: ",
              "%0.5f - %s: %0.5f - val_loss: %0.5f - %s: %0.5f" %
              (init_train_loss, self.train_acc_str, init_train_acc,
               init_test_loss, self.val_acc_str, init_test_acc))

        return (init_train_loss, init_train_acc,
               init_test_loss, init_test_acc)

            
    def train(self, data_augmentation=True):

        checkpoint_dir = os.path.join(self.expt_dir,"checkpoints","checkpoint")
        results_path = os.path.join(self.expt_dir, 'results.txt')
        fig_path = [os.path.join(self.expt_dir, 'results_acc.png'),
                    os.path.join(self.expt_dir, 'results_loss.png')]
        json_path = os.path.join(self.expt_dir, 'results.json')
    
        # Create data_generator - if necessary
        dm = self.data_manager
        if not dm.data_generator_module:
            # No data generator used
            print ("No data augmentation")
            datagen = dummy_datagen = None
        else:
            # fit the model on the batches generated by datagen.flow()
            print ("Loading data generator")
            train_generator_dict = {**self.preprocess_param_dict, 
                                    **self.augment_param_dict}
            val_generator_dict = self.preprocess_param_dict

            datagen = ImageDataGenerator(**train_generator_dict)

            datagen.fit(dm.X_train)
            datagen = datagen.flow(dm.X_train, dm.Y_train, batch_size=dm.batch_size)
            datagen.numImages = dm.Y_train.shape[0]
            datagen.batches_per_epoch = datagen.numImages //dm.batch_size

            dummy_datagen = ImageDataGenerator(**val_generator_dict)
            dummy_datagen.fit(dm.X_test)
            dummy_datagen = dummy_datagen.flow(dm.X_test, dm.Y_test, batch_size=dm.batch_size)
            dummy_datagen.numImages = dm.Y_test.shape[0]
            dummy_datagen.batches_per_epoch = dummy_datagen.numImages // dm.batch_size


        (init_train_loss, init_train_acc,
               init_test_loss, init_test_acc) = self.get_init_condits(datagen,
                                                                      dummy_datagen)

        training_monitor = TrainingMonitor(fig_path, jsonPath=json_path,
                                           resultsPath = results_path)
        callbacks = [training_monitor]
        
        checkpointer = ModelCheckpoint(checkpoint_dir,
                                       monitor = self.val_acc_str,
                                       verbose=1,
                                       data_manager=self.data_manager,
                                       period=self.epochs_per_recording,
                                       nocheckpoint = self.nocheckpoint)
        callbacks.append(checkpointer)

        # Add lr scheduler
        if self.lr_schedule:
            lr_scheduler = StepLearningRateScheduler(self.lr_schedule)
            callbacks.append(lr_scheduler)



        # Get results for initialized net
        training_monitor.on_train_begin()
        log_dir = {'loss': init_train_loss,
                   'val_loss': init_test_loss}
        log_dir[self.train_acc_str] = init_train_acc
        log_dir[self.val_acc_str] = init_test_acc
        
        training_monitor.on_epoch_end(epoch=0,logs = log_dir)
        checkpointer.set_model(self.model)
        checkpointer.on_epoch_end(epoch=-1,logs = log_dir)

        # Train Model
        dm = self.data_manager
        if not dm.data_generator_module:
            #print ("No data augmentation")
            results = self.model.fit(dm.X_train,
                                     dm.Y_train,
                                     batch_size=128,#dm.batch_size,
                                     epochs=self.epochs,
                                     validation_data=(dm.X_test,
                                                      dm.Y_test),
                                     callbacks=callbacks,
                                     shuffle=True)
            
        else:
            # fit the model on the batches generated by datagen.flow()
            #print ("Loading data generator")
            #train_generator_dict = {**self.preprocess_param_dict, 
            #                        **self.augment_param_dict}
            #val_generator_dict = self.preprocess_param_dict

            #datagen = ImageDataGenerator(**train_generator_dict)

            #datagen.fit(dm.X_train)
            #datagen = datagen.flow(dm.X_train, dm.Y_train, batch_size=dm.batch_size)
            #datagen.numImages = dm.Y_train.shape[0]
            #datagen.batches_per_epoch = datagen.numImages //dm.batch_size

            #dummy_datagen = ImageDataGenerator(**val_generator_dict)
            #dummy_datagen.fit(dm.X_test)
            #dummy_datagen = dummy_datagen.flow(dm.X_test, dm.Y_test, batch_size=dm.batch_size)
            #dummy_datagen.numImages = dm.Y_test.shape[0]
            #dummy_datagen.batches_per_epoch = dummy_datagen.numImages // dm.batch_size

            results = self.model.fit_generator(datagen,
                                       steps_per_epoch=\
                                           datagen.batches_per_epoch,
                                       epochs=self.epochs,
                                       validation_data=dummy_datagen,
                                       validation_steps=\
                                           dummy_datagen.batches_per_epoch,
                                       callbacks = callbacks,
                                       shuffle=True)

        self.best_score = checkpointer.best_score
        self.best_epoch = checkpointer.best_epoch
        
