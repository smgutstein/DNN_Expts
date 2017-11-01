from __future__ import absolute_import
from keras import backend as K
import local_backend as LK
import numpy as np
from theano import config
from theano import tensor as T
from theano import function as Tfunc
from theano import scan as theano_scan
from theano import In

def binary_accuracy_local(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def hot_bit(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


def nearest_target(x, targets):
    # Generate Cartesian Product of rows to compare
    # i = T.ivector('i')
    # j = T.ivector('j')

    i_values = T.arange(x.shape[0]).repeat(targets.shape[0])  # outer rows/loop
    j_values = T.tile(T.arange(targets.shape[0]), x.shape[0])  # inner rows/loop

    # Get dists between each pair of rows in x & targets
    dists, updates = theano_scan(lambda ii, jj, xx, yy: T.sum((xx[ii, :] - yy[jj, :])**2),
                                 sequences=[i_values, j_values], non_sequences=[x, targets])
    
    # Get indices of rows in targets that are closest to each row of x
    indices = T.flatten(T.argmin(dists.reshape((x.shape[0], targets.shape[0])), axis=-1))

    # Recover each targets-row from the indices
    nn, updates = theano_scan(lambda idx, targets: targets[idx, :],
                              sequences=[indices],
                              non_sequences=targets) 

    return nn

def ECOC_accuracy(y_encode):
    def calc_acc(y_true, y_pred):
        acc_list, updates = theano_scan(fn=lambda a, b: LK.allclose(a, b),
                                        sequences=[y_true, y_pred])
        return LK.true_div(acc_list.sum(), acc_list.size)
    
    def ECOC_fnc(y_true, y_pred):
        return K.cast(K.mean(calc_acc(y_true,
                      nearest_target(y_pred, y_encode))),
                      K.floatx())
    return ECOC_fnc



def ECOC_fast_accuracy(y_encode):
    Y_Encoding = T.constant(y_encode, dtype=K.floatx())
    def ECOC_fnc(Y_True, Y_Pred):
        # Find nearest trgt vector to actual output
        deltas = Y_Pred.reshape((Y_Pred.shape[0], 1, -1)) - Y_Encoding.reshape((1, Y_Encoding.shape[0], -1))
        dists = T.sum(T.pow(deltas, 2), axis=-1)
        nearest_trgt = Y_Encoding[T.argmin(dists, axis=-1)]

        # See if 'best guess' trgt vector is correct
        dists2 = T.sum(T.pow(Y_True - nearest_trgt,2), axis=-1)
        acc = T.true_div(T.prod(T.shape((dists2 < 1e-06).nonzero())),T.prod(T.shape(dists2)))

        return K.cast(acc, K.floatx())
        
    return ECOC_fnc
