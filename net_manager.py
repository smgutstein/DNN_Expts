from __future__ import print_function

import errno
import importlib
import inspect
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import model_from_json, model_from_yaml
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
                 batch_size=32,
                 data_augmentation=True,
                 save_iters=True):

        self.init_epoch = 0
        self.epochs = int(expt_param_dict['epochs'])
        self.data_manager = data_manager
        self.expt_dir = expt_dir
        self.expt_prefix = os.path.basename(expt_dir)
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation
        self.save_iters = save_iters
        if self.save_iters:
            self.epochs_per_recording = int(expt_param_dict['epochs_per_recording'])
        else:
            self.epochs_per_recording = self.epochs

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
        self.model.compile(loss='mean_squared_error',
                           optimizer=self.opt,
                           metrics=[metric_fnc])

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
            return build_architecture(input_shape,
                                      self.nb_output_nodes,
                                      net_param_dict['output_activation'])

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

    def train(self, data_augmentation=True, batch_size=32):

        results_file = open(os.path.join(self.expt_dir, 'results.txt'), 'a')
        init_train_loss, init_train_acc = \
        self.model.evaluate(self.data_manager.X_train,
                            self.data_manager.Y_train)
        init_test_loss, init_test_acc = \
        self.model.evaluate(self.data_manager.X_test,
                            self.data_manager.Y_test)
        print("\nInit loss and acc:                             loss: ",
              "%0.5f - acc: %0.5f - val_loss: %0.5f - val_acc: %0.5f" %
              (init_train_loss, init_train_acc,
               init_test_loss, init_test_acc))

        # Record initial responses before any training as epoch 0
        epoch_str = 'Epoch ' + str(self.init_epoch) + ':  '
        results_str1 = 'Train Acc: {:5.4f}  Train Loss {:5.4f}'.format(init_train_acc, init_train_loss)
        results_str2 = 'Test Acc {:5.4f}  Test Loss {:5.4f}\n'.format(init_test_acc, init_test_loss)
        results_str = epoch_str + '  '.join([results_str1, results_str2])
        results_file.write(results_str)
        self.save_net(self.init_epoch)
        self.init_epoch += 1

        for rec_num in range(self.epochs/self.epochs_per_recording):
            # Train Model
            if not data_augmentation:
                print('Not using data augmentation.')
                dm = self.data_manager
                results = self.model.fit(dm.X_train,
                                         dm.Y_train,
                                         batch_size=batch_size,
                                         epochs=self.epochs_per_recording,
                                         steps_per_epoch=dm.X_train.shape[0] // batch_size,
                                         validation_data=(dm.X_test,
                                                          dm.Y_test),
                                         shuffle=True)
            else:
                print('Using real-time data augmentation.')

                # this will do preprocessing and realtime data augmentation
                datagen = ImageDataGenerator(
                    featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=True,  # randomly flip images
                    vertical_flip=False)  # randomly flip images

                # compute quantities required for featurewise
                # normalization
                # (std, mean, and principal components
                #  if ZCA whitening is applied)
                dm = self.data_manager
                datagen.fit(dm.X_train)

                # fit the model on the batches generated by datagen.flow()
                results = self.model.fit_generator(datagen.flow(dm.X_train,
                                                                dm.Y_train,
                                                                batch_size=batch_size),
                                                   steps_per_epoch=(dm.X_train.shape[0] //
                                                                    self.batch_size),
                                                   epochs=self.epochs_per_recording,
                                                   validation_data=(dm.X_test, dm.Y_test))

            rh = results.history

            # Assumes only two metrics are 'loss' and name of accuracy metric
            temp = set(self.model.metrics_names) - set(['loss'])
            tr_acc_name = temp.pop()
            va_acc_name = 'val_' + tr_acc_name
            
            for ctr, (tr_acc, tr_loss, te_acc, te_loss) in enumerate(zip(rh[tr_acc_name],
                                                                         rh['loss'],
                                                                         rh[va_acc_name],
                                                                         rh['val_loss'])):
                epoch_str = 'Epoch ' + str(ctr + self.init_epoch + rec_num*self.epochs_per_recording) + ':  '
                results_str1 = 'Train Acc: {:5.4f}  Train Loss {:5.4f}'.format(tr_acc, tr_loss)
                results_str2 = 'Test Acc {:5.4f}  Test Loss {:5.4f}\n'.format(te_acc, te_loss)
                results_str = epoch_str + '  '.join([results_str1, results_str2])
                results_file.write(results_str)

            if self.save_iters:
                epoch_num = str((rec_num+1)*self.epochs_per_recording + (self.init_epoch-1))
                self.save_net(epoch_num)
        results_file.close()

    def save_net(self, epoch_num):
        # Save net weights
        wt_file_name = self.expt_prefix + "_weights_" + \
                       str(epoch_num) + ".h5"
        weights_file = os.path.join(self.expt_dir, wt_file_name)

        dm = self.data_manager

        if 'temp' in self.expt_dir:
            overwrite = True
        else:
            overwrite = False
        print("Saving ", weights_file)
        self.model.save_weights(weights_file, overwrite=overwrite)

        if (dm.encoding_dict == dm.curr_encoding_info['encoding_dict'] and
            dm.label_dict == dm.curr_encoding_info['label_dict']):

            print("No encoding change")

        else:
            dm.curr_encoding_info['encoding_dict'] = dm.encoding_dict
            dm.curr_encoding_info['label_dict'] = dm.label_dict
            dm.curr_encoding_info['meta_encoding_dict'] = dm.meta_encoding_dict

            encodings_file_name = self.expt_prefix + '_encodings_' + \
                str(epoch_num) + '.pkl'
            print ("Saving", encodings_file_name)
            with open(os.path.join(self.expt_dir, encodings_file_name), 'w') as f:
                pickle.dump(dm.curr_encoding_info, f)
