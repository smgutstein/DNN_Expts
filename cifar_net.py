from __future__ import print_function

import importlib
import inspect
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import model_from_json, model_from_yaml
import os
import pickle

from data_manager import DataManager

class Cifar_Net(object):

    def __init__(self, data_manager,
                 expt_dir,
                 net_param_dict,
                 expt_param_dict,
                 metric_param_dict,
                 optimizer_param_dict,
                 batch_size=32,
                 data_augmentation=True,
                 save_iters=True):

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
        self.init_data_manager(data_manager)

        # Import accuracy function
        temp = importlib.import_module(metric_param_dict['metrics_module'])
        metric_fnc = getattr(temp, metric_param_dict['accuracy_metric'])
        metric_fnc_args = inspect.getargspec(metric_fnc)
        if metric_fnc_args.args == ['y_encode']:
            metric_fnc = metric_fnc(self.data_manager.encoding_matrix)
        self.acc_metric = metric_param_dict['accuracy_metric']

        print("Initializing architecture ...")
        self.init_model(net_param_dict)

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

    def init_data_manager(self, data_manager):
        self.data_manager = data_manager

    def init_model(self, net_param_dict):
        
        # Import accuracy function
        temp = importlib.import_module(net_param_dict['arch_module'])
        build_architecture = getattr(temp, "build_architecture")
        if K.image_data_format() != 'channels_last':
            input_shape = (self.data_manager.img_channels,
                           self.data_manager.img_rows,
                           self.data_manager.img_cols)
        else:
            input_shape = (self.data_manager.img_rows,
                           self.data_manager.img_cols,
                           self.data_manager.img_channels)

        if 'saved_arch' in net_param_dict:
            # Load architecture
            if net_param_dict['saved_arch'][-4:] == 'json':
                with open(net_param_dict['saved_arch'], 'r') as f:
                    json_str = f.read()
                    self.model = model_from_json(json_str)
            elif net_param_dict['saved_arch'][-4:] == 'yaml':
                with open(net_param_dict['saved_arch'], 'r') as f:
                    yaml_str = f.read()
                    self.model = model_from_yaml(yaml_str)

            # Load weights
            self.model.load_weights(net_param_dict['saved_weights'])
        else:
            self.model = build_architecture(input_shape,
                                            self.nb_output_nodes,
                                            net_param_dict['output_activation'])

            # Save net architecture, weights and class/encoding info
            json_str = self.model.to_json()        
            model_file = os.path.join(self.expt_dir,
                                      self.expt_prefix + "_init.json")
            open(model_file, "w").write(json_str)

            # Save initial net weights
            self.save_net("0")

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
                epoch_str = 'Epoch ' + str(ctr + rec_num*self.epochs_per_recording) + ':  '
                results_str1 = 'Train Acc: {:5.4f}  Train Loss {:5.4f}'.format(tr_acc, tr_loss)
                results_str2 = 'Test Acc {:5.4f}  Test Loss {:5.4f}\n'.format(te_acc, te_loss)
                results_str = epoch_str + '  '.join([results_str1, results_str2])
                results_file.write(results_str)

            if self.save_iters:
                epoch_num = str((rec_num + 1)*self.epochs_per_recording)
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
            pickle.dump(dm.curr_encoding_info,
                        open(os.path.join(self.expt_dir, encodings_file_name), 'w'))


if __name__ == '__main__':

    import pdb
    pdb.set_trace()
    x = DataManager('n_hot_encoding', 10)
    x.make_encoding_dict(nb_hot=1)
    x.encode_labels()
    y = Cifar_Net(10, x, 'temp', 'expt1')
    import pdb
    pdb.set_trace()
    y.train()
