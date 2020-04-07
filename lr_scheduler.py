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

'''
class LR_Scheduler(Callback):

    def __init__(self, opt, lr_schedule):
        self.opt = opt
        self.lr_schedule = lr_schedule

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % 2 != 0:
            new_val = epoch*.3+.3
        else:
            new_val = epoch*.2+.2

        #self.opt.delta = tf.constant(new_val, dtype = self.opt.lr.dtype)
        SGD_VAR.learn_rate = new_val
        print ("New LR: ", new_val)
'''
'''
      #Copying graph node that stores original value of learning rate
        lr = self.lr 

        # Checking whether learning rate schedule is to be used
        if self.initial_lr_decay > 0:
            # this decay mimics exponential decay from 
            # tensorflow/python/keras/optimizer_v2/exponential_decay 

            # Get value of current number of processed batches from graph node
            # and convert to numeric value for use in K.pow()
            curr_batch = float(K.get_value(self.iterations))

            # Create graph node containing lr decay factor
            # Note: self.lr_decay_steps is a number, not a node
            #       self.lr_decay is a node, not a number
            decay_factor =  K.pow(self.lr_decay, (curr_batch / self.lr_decay_steps)) 

            # Reassign lr to graph node formed by
            # product of graph node containing decay factor
            # and graph node containing original learning rate.
            lr = lr * decay_factor

            # Get product of two numbers to calculate number of batches processed
            # in warmup period
            num_warmup_batches = self.steps_per_epoch_num * self.warmup_epochs

            # Make comparisons between numbers to determine if we're in warmup period
            if (self.warmup_epochs > 0) and (curr_batch < num_warmup_batches):

                # Create node with value of learning rate by multiplying a number
                # by a node, and then dividing by a number
                lr = (self.initial_lr  *
                      K.cast(self.iterations, K.floatx()) / curr_batch)
'''



class LRScheduleFunction(Callback):
    def __init__(self, sched_function,
                 on_batch=False, on_epoch=False,):
        self.sched_function = sched_function
        self.epoch = 0
        self.batch = 0
        self.on_batch = on_batch
        self.on_epoch = on_epoch

    def on_batch_begin(self, batch, logs=None):
        self.batch += 1
        if self.on_batch:
          lr = self.sched_function(self.batch, self.epoch)
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


        
