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

def nearest_neighbor_vec(x, y):
    '''Find nearest output code vector
       for each nn output'''

    #Generate Cartesian Product of rows to compare
    i = T.ivector('i')
    j = T.ivector('j')

    i_values = T.arange(x.shape[0]).repeat(y.shape[0]) #outer rows/loop
    j_values = T.tile(T.arange(y.shape[0]), x.shape[0]) #inner rows/loop

    #Get dists between each pair of rows in x & y
    res,updates = theano.scan(lambda ii, jj, xx, yy: T.sum((xx[ii,:] - yy[jj,:])**2), 
                              sequences=[i_values ,j_values], non_sequences=[x, y])
    #Get indices of rows in y that are closest to each row of x
    indices = T.flatten(T.argmin(res.reshape((x.shape[0], y.shape[0])), axis = -1))

    #Recover each y-row from the indices
    res2, updates = theano.scan(lambda idx, y: y[idx,:], sequences=[indices], non_sequences=y) 

    return res2


# NN OPERATIONS

def _assert_has_capability(module, func):
    if not hasattr(module, func):
        raise EnvironmentError(
            'It looks like like your version of '
            'Theano is out of date. '
            'Install the latest version with:\n'
            'pip install git+git://github.com/Theano/Theano.git '
            '--upgrade --no-deps')


def elu(x, alpha=1.0):
    """ Exponential linear unit

    # Arguments
        x: Tensor to compute the activation function for.
        alpha: scalar
    """
    _assert_has_capability(T.nnet, 'elu')
    return T.nnet.elu(x, alpha)


def relu(x, alpha=0., max_value=None):
    _assert_has_capability(T.nnet, 'relu')
    x = T.nnet.relu(x, alpha)
    if max_value is not None:
        x = T.minimum(x, max_value)
    return x


def softmax(x):
    return T.nnet.softmax(x)


def softplus(x):
    return T.nnet.softplus(x)


def softsign(x):
    return T_softsign(x)


def categorical_crossentropy(target, output, from_logits=False):
    if from_logits:
        output = T.nnet.softmax(output)
    else:
        # scale preds so that the class probas of each sample sum to 1
        output /= output.sum(axis=-1, keepdims=True)
    # avoid numerical instability with _EPSILON clipping
    output = T.clip(output, epsilon(), 1.0 - epsilon())
    return T.nnet.categorical_crossentropy(output, target)


def sparse_categorical_crossentropy(target, output, from_logits=False):
    target = T.cast(T.flatten(target), 'int32')
    target = T.extra_ops.to_one_hot(target, nb_class=output.shape[-1])
    target = reshape(target, shape(output))
    return categorical_crossentropy(target, output, from_logits)


def binary_crossentropy(target, output, from_logits=False):
    if from_logits:
        output = T.nnet.sigmoid(output)
    # avoid numerical instability with _EPSILON clipping
    output = T.clip(output, epsilon(), 1.0 - epsilon())
    return T.nnet.binary_crossentropy(output, target)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)


def tanh(x):
    return T.tanh(x)


def dropout(x, level, noise_shape=None, seed=None):
    """Sets entries in `x` to zero at random,
    while scaling the entire tensor.

    # Arguments
        x: tensor
        level: fraction of the entries in the tensor
            that will be set to 0.
        noise_shape: shape for randomly generated keep/drop flags,
            must be broadcastable to the shape of `x`
        seed: random seed to ensure determinism.
    """
    if level < 0. or level >= 1:
        raise ValueError('Dropout level must be in interval [0, 1[.')
    if seed is None:
        seed = np.random.randint(1, 10e6)
    if isinstance(noise_shape, list):
        noise_shape = tuple(noise_shape)

    rng = RandomStreams(seed=seed)
    retain_prob = 1. - level

    if noise_shape is None:
        random_tensor = rng.binomial(x.shape, p=retain_prob, dtype=x.dtype)
    else:
        random_tensor = rng.binomial(noise_shape, p=retain_prob, dtype=x.dtype)
        random_tensor = T.patternbroadcast(random_tensor,
                                           [dim == 1 for dim in noise_shape])
    x *= random_tensor
    x /= retain_prob
    return x


def l2_normalize(x, axis=None):
    square_sum = T.sum(T.square(x), axis=axis, keepdims=True)
    norm = T.sqrt(T.maximum(square_sum, epsilon()))
    return x / norm


def in_top_k(predictions, targets, k):
    """Returns whether the `targets` are in the top `k` `predictions`.

    # Arguments
        predictions: A tensor of shape `(batch_size, classes)` and type `float32`.
        targets: A 1D tensor of length `batch_size` and type `int32` or `int64`.
        k: An `int`, number of top elements to consider.

    # Returns
        A 1D tensor of length `batch_size` and type `bool`.
        `output[i]` is `True` if `predictions[i, targets[i]]` is within top-`k`
        values of `predictions[i]`.
    """
    # handle k < 1 and k >= predictions.shape[1] cases to match TF behavior
    if k < 1:
        # dtype='bool' is only available since Theano 0.9.0
        try:
            return T.zeros_like(targets, dtype='bool')
        except TypeError:
            return T.zeros_like(targets, dtype='int8')

    if k >= int_shape(predictions)[1]:
        try:
            return T.ones_like(targets, dtype='bool')
        except TypeError:
            return T.ones_like(targets, dtype='int8')

    predictions_k = T.sort(predictions)[:, -k]
    targets_values = predictions[T.arange(targets.shape[0]), targets]
    return T.ge(targets_values, predictions_k)
