from collections import defaultdict
from contextlib import contextmanager
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
# Legacy functions
from keras.backend.common import set_image_dim_ordering, image_dim_ordering

py_all = all
py_sum = sum


# INTERNAL UTILS
theano.config.floatX = floatx()
_LEARNING_PHASE = T.scalar(dtype='uint8', name='keras_learning_phase')  # 0 = test, 1 = train
_UID_PREFIXES = defaultdict(int)

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    '''Returns symbolic 'int8' value if all
       elements in tensors a & b are within
       given tolerances'''

    return T.allclose(a, b, rtol, atol, equal_nan)

def true_div(num, denom):
    return T.true_div(num, denom)

