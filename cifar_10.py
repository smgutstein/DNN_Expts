from __future__ import print_function

from collections import defaultdict
import copy

from keras.preprocessing.image import ImageDataGenerator
from keras.engine.training import _make_batches
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adagrad
from keras.utils import np_utils
from keras import backend as K

import json
import numpy as np
from operator import itemgetter
import os
import pickle
import random
import scipy
import sys

from theano import tensor as T
from theano import function as Tfunc

from data_manager_recon_cifar10 import DataManager

class Cifar_Net(object):

    # input image dimensions
    img_rows, img_cols = 32, 32
    
    # the CIFAR10 images are RGB
    img_channels = 3

    def __init__(self, epochs, data_manager,
                 expt_dir, expt_prefix,
                 batch_size = 32, data_augmentation = True,
                 epochs_per_recording = None,
                 save_iters = True):

        self.epochs = epochs
        self.data_manager = data_manager
        self.expt_dir = expt_dir
        self.expt_prefix = expt_prefix
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation
        if epochs_per_recording is None:
            self.epochs_per_recording = epochs
        else:
            self.epochs_per_recording = epochs_per_recording
        self.save_iters = save_iters

        # Ensure expt output dir exists
        if expt_dir is not None:
           try: 
             os.makedirs(expt_dir)
           except OSError:
             if not os.path.isdir(expt_dir):
                raise        
                
        # Make optimizer
        self.opt = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)


        # Prepare standard training
        print("Standard training")
        self.nb_output_nodes = data_manager.nb_code_bits
        print("Initializing data manager ...")
        self.init_data_manager(data_manager)
        print("Initializing architecture ...")
        self.init_model() 

        # Compile model
        print("Compiling model ...")
        self.model.compile(loss='mean_squared_error',
                           optimizer=self.opt,
                           metrics=['accuracy'])

        # Summarize            
        self.summary()



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
        #pickle.dump(self.data_manager,
        #            open(os.path.join(self.expt_dir, 'data_manager.pkl'), 'w'))


    def init_model(self):
        self.model = Sequential()

        if K.image_data_format() != 'channels_last':
            self.model.add(Convolution2D(32, (3, 3), padding='same',
                           input_shape=(self.img_channels,
                                        self.img_rows,
                                        self.img_cols)))
        else:
            self.model.add(Convolution2D(32, (3, 3), padding='same',
                           input_shape=(self.img_rows,
                                        self.img_cols,
                                        self.img_channels)))

        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.nb_output_nodes))
        self.model.add(Activation('softmax'))
        

        #if self.final_layer is not None and self.final_layer != '':
            #self.model.add(Activation(self.final_layer))
            #if self.final_layer == 'tanh':
            #    self.new_not_hot, self.new_hot = get_tanh_hot_not_hot()
            #    self.data_manager.reset_targets(self.new_hot, self.new_not_hot)

        # Save net architecture, weights and class/encoding info
        json_str = self.model.to_json()        
        model_file = os.path.join(self.expt_dir, self.expt_prefix + "_init.json")
        open(model_file,"w").write(json_str)

        # Save initial net weights
        self.save_net("0")

    def train(self, data_augmentation = True, batch_size = 32):

        init_train_loss, init_train_acc = self.model.evaluate(self.data_manager.X_train,
                                                              self.data_manager.Y_train)
        init_test_loss, init_test_acc = self.model.evaluate(self.data_manager.X_test,
                                                            self.data_manager.Y_test)
        print("Init loss and acc:                             loss: %0.5f - acc: %0.5f - val_loss: %0.5f - val_acc: %0.5f"%
              (init_train_loss, init_train_acc,init_test_loss, init_test_acc))

        for rec_num in range(self.epochs/self.epochs_per_recording):

            # Train Model
            if not data_augmentation:
                print('Not using data augmentation.')
                dm = self.data_manager
                self.model.fit(dm.X_train,
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

                # compute quantities required for featurewise normalization
                # (std, mean, and principal components if ZCA whitening is applied)
                dm = self.data_manager
                datagen.fit(dm.X_train)

                # fit the model on the batches generated by datagen.flow()
                self.model.fit_generator(datagen.flow(dm.X_train,
                                                      dm.Y_train,
                                                      batch_size=batch_size),
                                         steps_per_epoch=(dm.X_train.shape[0] //
                                                          self.batch_size),
                                         epochs=self.epochs_per_recording,
                                         validation_data=(dm.X_test,
                                                          dm.Y_test))

            if self.save_iters:
                epoch_num = str((rec_num + 1) * self.epochs_per_recording)
                self.save_net(epoch_num)


    def save_net(self, epoch_num):
        # Save net weights 
        weights_file = os.path.join(self.expt_dir,
                                    self.expt_prefix + "_weights_" + str(epoch_num) + ".h5")

        print ("Saving ", weights_file)
        if 'temp.txt' in weights_file:
            overwrite = True
        else:
            overwrite = False
        self.model.save_weights(weights_file, overwrite = True)


if __name__ == '__main__':

    x = DataManager(10)
    x.make_n_hot_encoding_dict()
    x.encode_labels()
    y = Cifar_Net(10, x, 'temp', 'expt1')
    y.train()
