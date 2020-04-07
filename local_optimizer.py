from __future__ import print_function

from keras import backend as K
from keras import optimizers
from keras.optimizers import SGD, RMSprop, Adam
from keras.legacy import interfaces
import sys
import tensorflow as tf


def sgd(sgd_params):
    # Stochastic gradient descent
    # e.g. SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    return optimizers.SGD(**sgd_params)

def sgd_schedule(sgd_params):
    # Stochastic gradient descent
    # e.g. SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    if "lr_schedule" in sgd_params:
        temp_params = {x:sgd_params[x] for x in sgd_params
                       if x != "lr_schedule"}

    return optimizers.SGD(**temp_params)

def sgd_var(sgd_params):
    # Stochastic gradient descent
    # e.g. SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    return SGD_VAR(**sgd_params)


def adam(adam_params):
    # ADAM - ADAptive Moment estimation
    # e.g. Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
    #             epsilon=None, decay=0., amsgrad=False)
    return optimizers.Adam(**adam_params)



class SGD_VAR(SGD):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.05, momentum=0., decay=0.,
                 nesterov=False, lr_dict= {},  **kwargs):

        super(SGD_VAR, self).__init__(lr, momentum, decay,
                                      nesterov, **kwargs)

        # Initializing learning rate schedule dict
        if lr_dict == {}:
            self.lr_dict = {0:lr}
        else:
            self.lr_dict = lr_dict
            

        with K.name_scope(self.__class__.__name__):
            self.iterations_ref = K.variable(0, dtype='int64', name='iterations_ref')
            self.new_lr = K.variable(lr, name='new_lr')

    def set_batches_per_epoch(self, batches_per_epoch=1562):
        self.batches_per_epoch = batches_per_epoch
        

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        # Adapted from SGD code, but with changes to allow adjustments in learning rate
        
        def lr_stepper(iteration, lr):
            ''' Wrapped python method used by tensor to determine desired learning rate'''
        
            # Change the learning rate where specified by lr_dict
            for x in self.lr_dict:
                temp = tf.Variable((x-1) * self.batches_per_epoch, dtype=iteration.dtype)
                if tf.equal(temp, iteration):
                    return tf.constant(self.lr_dict[x], dtype=lr.dtype)

            return lr

        # NOTE: K.update_add and K.update return tf.assign_add and tf.assign, respectively
        self.updates = [K.update_add(self.iterations, 1)]


        # Key lines to change self.lr
        new_lr = tf.contrib.eager.py_func(func=lr_stepper, inp=[self.iterations,
                                                                self.lr], Tout=tf.float32)

        new_iter_ref = tf.cond(tf.math.equal(self.lr,new_lr),
                               lambda: K.update_add(self.iterations_ref, 1),
                               lambda: K.update(self.iterations_ref, 1))
        self.updates.append(K.update(self.lr, new_lr))
        self.updates.append(new_iter_ref)
        
        # Temporary code to debug output
        #self.iterations = tf.Print(self.lr,
        #         [self.iterations,self.iterations_ref, self.lr],
        #                           message="\n Debug Vals:" )

        grads = self.get_gradients(loss, params)
        
        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]

        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates


class EffNet_RMSprop(Optimizer):
    """RMSProp optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).

    This optimizer is usually a good choice for recurrent
    neural networks.

    # Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.

    Following tensorflow implementation in tf_rmsprop.py & training_ops_gpu.cu.cc
    mean_square = decay * mean_square{t-1} + (1-decay) gradient**2
    mom = momentum * mom{t-1} + learning_rate * gradient /sqrt(mean_square + epsilon)
    delta = - mpm

    # References
        - [rmsprop: Divide the gradient by a running average of its recent magnitude
           ](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    """

    def __init__(self, base_lr, train_batch_size, steps_per_epoch,
                 momentum, rho, lr_decay_steps, lr_decay=1, 
                 warmup_epochs=0, epsilon=None,
                 **kwargs):        
        super(RMSprop, self).__init__(**kwargs)
        
        with K.name_scope(self.__class__.__name__):
            # From efficientnet main.py line 387
            scaled_lr = base_lr * train_batch_size / 256.0
            
            self.lr = K.variable(scaled_lr, name='lr')
            self.steps_per_epoch = K.variable(steps_per_epoch,
                                               dtype='int32',
                                               name='steps_per_epoch')
            self.momentum = K.variable(momentum, name='momentum')
            self.rho = K.variable(rho, name='rho')
            self.lr_decay = K.variable(lr_decay, name='lr_decay')
            self.iterations = K.variable(0, dtype='int32', name='iterations')
            
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_lr = scaled_lr
        self.initial_lr_decay = lr_decay
        self.lr_decay_steps = lr_decay_steps
        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch_num = steps_per_epoch
        
    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)

        # Set lr: Start w/initial lr, account for decay due batches processed
        #         and account for warm-up epochs
        # Should eventually be replaced with LearningRateScheduler, but this schedule
        # updates as a function of batch, not epoch.
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
        # Old Code - To Be Deleted
        lr = self.lr
        if self.initial_lr_decay > 0:
            # mimics exponential decay from tensorflow/python/keras/optimizer_v2/exponential_decay
            # which is called tf.train.exponential_decay in efficientnet/utils.py
            curr_batch = float(K.get_value(self.iterations))
            decay_factor =  K.pow(self.lr_decay, (curr_batch / self.lr_decay_steps)) 
            lr = lr * decay_factor
            num_warmup_batches = self.steps_per_epoch_num * self.warmup_epochs
            if (self.warmup_epochs > 0) and (curr_batch < num_warmup_batches):
                warmup_steps = self.warmup_epochs * self.steps_per_epoch
                warmup_lr = (self.initial_lr  *
                             K.cast(self.iterations, K.floatx()) / K.cast(warmup_steps, K.floatx()))
                lr =  warmup_lr
        '''
        # Updates
        self.updates = [K.update_add(self.iterations, 1)]

        # Initialize accumulators used to update each parameter
        mean_sq_accs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        mom_accs =  [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        # Set weights used to restart optimizer in current state
        self.weights = [self.iterations] + mean_sq_accs + mom_accs

        # Update Parameters
        for p, g, ms, ma in zip(params, grads, mean_sq_accs, mom_accs):
            # RMS_Prop calculations
            # (following tensorflow/core/kernels/training_ops.cc::ApplyRMSProp)
            new_ms = self.rho * ms + (1. - self.rho) * K.square(g)
            new_mom = self.momentum * ma + lr * g / (K.sqrt(new_ms + self.epsilon))
            new_p = p - new_mom

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            # RMS_Prop Updates
            self.updates.append(K.update(ms, mean_sq_accs))
            self.updates.append(K.update(ma, mom_accs))
            self.updates.append(K.update(p, new_p))
        
        return self.updates

    
    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'steps_per_epoch': float(K.get_value(self.steps_per_epoch)),
                  'momentum': float(K.get_value(self.momentum)),
                  'rho': float(K.get_value(self.rho)),
                  'lr_decay': float(K.get_value(self.lr_decay)),
                  'warmup_epochs': self.warmup_epochs,
                  'iterations': int(K.get_value(self.iterations)),
                  'initial_lr_decay': float(self.initial_lr_decay),
                  'lr_decay_steps': float(self.lr_decay_steps),
                  'epsilon': self.epsilon}
        base_config = super(RMSprop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
