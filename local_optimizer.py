from __future__ import print_function

from keras import optimizers
from keras import backend as K
from keras import optimizers
from keras.optimizers import SGD
from keras.legacy import interfaces
import sys
import tensorflow as tf


def sgd(sgd_params):
    # Stochastic gradient descent
    # e.g. SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    return optimizers.SGD(**sgd_params)

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
