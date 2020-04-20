from __future__ import print_function

from keras.callbacks import BaseLogger, Callback
from keras import backend as K
from local_optimizer import SGD_VAR
import matplotlib.pyplot as plt
import numpy as np
import json
from operator import itemgetter
import os
import pickle
import shutil
import tensorflow as tf


class LRScheduleFunction(Callback):
    def __init__(self, sched_function, on_batch,
                 on_epoch, kwargs):
        self.epoch = 0
        self.batch = 0
        self.on_batch = on_batch
        self.on_epoch = on_epoch
        self.sched_function = sched_function
        self.kwargs = kwargs

    def on_batch_begin(self, batch, logs=None):
        self.batch += 1
        if self.on_batch:
          self.kwargs["batch"] = self.batch
          self.kwargs["epoch"] = self.epoch
          lr = self.sched_function(self.kwargs)
          K.set_value(self.model.optimizer.lr, lr)
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch += 1
        if self.on_epoch:
          lr = self.sched_function(self.batch, self.epoch)
          K.set_value(self.model.optimizer.lr, lr)
          
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        # Cast is to ensure lr is json serializable. For some reason np.float32
        # is not serializable, but np.float64 is
        logs['lr'] = K.get_value(self.model.optimizer.lr).astype(np.float64)
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Cast is to ensure lr is json serializable. For some reason np.float32
        # is not serializable, but np.float64 is
        logs['lr'] = K.get_value(self.model.optimizer.lr).astype(np.float64)
        
class StepLearningRateScheduler(Callback):
    def __init__(self, schedule_list, verbose = 1): 
        super(StepLearningRateScheduler, self).__init__()
        self.schedule_list = schedule_list
        self.verbose = verbose

        # Ensure tuples in schedule_list are sorted with lr's
        # for earlier epochs at front of list
        self.schedule_list.sort(key = itemgetter(0))
        if self.schedule_list[0][0] != 0:
            raise ValueError('lr schedule does not define lr for epoch 0')
        else:
            _, lr = self.schedule_list.pop(0)
            
            if not isinstance(lr, (float, np.float32, np.float64)):
                raise ValueError('The output of the "schedule" function ' 'should be float.')

            self.lr = lr
            #K.set_value(self.model.optimizer.lr, lr)
            #import pdb
            #pdb.set_trace()

            print('\nInitial Learning Rate: %s' % (str(lr)))
            
    
    def on_epoch_begin(self, epoch, logs=None): 
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        if len(self.schedule_list) > 0 and epoch+1 >= self.schedule_list[0][0]:
            old_lr = self.lr
            _, lr = self.schedule_list.pop(0)
            
            if not isinstance(lr, (float, np.float32, np.float64)):
                raise ValueError('The output of the "schedule" function ' 'should be float.')

            self.lr = lr
            K.set_value(self.model.optimizer.lr, lr)

            if self.verbose > 0:
                print('\nEpoch %05d: StepLearningRateScheduler changing learning ' 'rate to %s. from %s' % (epoch + 1,
                                                                                                            str(lr),
                                                                                                            str(old_lr)))


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Cast is to ensure lr is json serializable. For some reason np.float32
        # is not serializable, but np.float64 is
        logs['lr'] = K.get_value(self.model.optimizer.lr).astype(np.float64)


        


class FactorLearningRateScheduler(Callback):
    def __init__(self, schedule_list, verbose = 1): 
        super(FactorLearningRateScheduler, self).__init__()
        self.schedule_list = schedule_list
        self.verbose = verbose

        # Ensure tuples in schedule_list are sorted with lr's
        # for earlier epochs at front of list
        self.schedule_list.sort(key = itemgetter(0))
        if self.schedule_list[0][0] != 0:
            raise ValueError('lr schedule does not define lr for epoch 0')
        else:
            _, lr = self.schedule_list.pop(0)
            
            if not isinstance(lr, (float, np.float32, np.float64)):
                raise ValueError('The output of the "schedule" function ' 'should be float.')

            self.lr = lr
            #K.set_value(self.model.optimizer.lr, lr)
            #import pdb
            #pdb.set_trace()

            print('\nInitial Learning Rate: %s' % (str(lr)))
            
    
    def on_epoch_begin(self, epoch, logs=None): 
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        if len(self.schedule_list) > 0 and epoch+1 >= self.schedule_list[0][0]:
            old_lr = self.lr
            _, lr_scale_factor = self.schedule_list.pop(0)
            
            if not isinstance(lr, (float, np.float32, np.float64)):
                raise ValueError('The output of the "schedule" function ' 'should be float.')

            self.lr = self.lr*lr_scale_factor
            K.set_value(self.model.optimizer.lr, self.lr)

            if self.verbose > 0:
                print('\nEpoch %05d: StepLearningRateScheduler changing learning ' 'rate to %s. from %s' % (epoch + 1,
                                                                                                            str(self.lr),
                                                                                                            str(old_lr)))


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Cast is to ensure lr is json serializable. For some reason np.float32
        # is not serializable, but np.float64 is
        logs['lr'] = K.get_value(self.model.optimizer.lr).astype(np.float64)


        
