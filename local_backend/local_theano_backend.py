from collections import defaultdict
from contextlib import contextmanager
from keras import backend as K
import theano
from theano import tensor as T
from theano import function as Tfunc
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal import pool
from theano.printing import Print
import scipy

try:
    import theano.sparse as th_sparse_module
except ImportError:
    th_sparse_module = None
try:
    from theano.tensor.nnet.nnet import softsign as T_softsign
except ImportError:
    from theano.sandbox.softsign import softsign as T_softsign

import numpy as np
from keras.backend.common import floatx, epsilon, image_data_format
from keras.utils.generic_utils import has_arg


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    '''Returns symbolic 'int8' value if all
       elements in tensors a & b are within
       given tolerances'''

    return T.allclose(a, b, rtol, atol, equal_nan)

def true_div(num, denom):
    return T.true_div(num, denom)


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



