from __future__ import absolute_import
from keras import backend as K
import local_backend as LK
from theano import tensor as T
from theano import scan as theano_scan

def binary_accuracy_local(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def hot_bit(y_true, y_pred):
    # Equivalent to categorical accuracy function in
    # keras metrics module
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())

def ECOC_fast_accuracy(y_encode):
    Y_Encoding = T.constant(y_encode, dtype=K.floatx())

    def ECOC_fnc(Y_True, Y_Pred):
        # Find nearest trgt vector to actual output
        deltas = Y_Pred.reshape((Y_Pred.shape[0], 1, -1)) - Y_Encoding.reshape((1, Y_Encoding.shape[0], -1))
        dists = T.sum(T.pow(deltas, 2), axis=-1)
        nearest_trgt = Y_Encoding[T.argmin(dists, axis=-1)]

        # See if 'best guess' trgt vector is correct
        dists2 = T.sum(T.pow(Y_True - nearest_trgt, 2), axis=-1)
        acc = T.true_div(T.prod(T.shape((dists2 < 1e-06).nonzero())), T.prod(T.shape(dists2)))

        return K.cast(acc, K.floatx())
        
    return ECOC_fnc
