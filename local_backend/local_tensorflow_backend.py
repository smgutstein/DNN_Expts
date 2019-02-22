from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import scipy
from keras import backend as K
import tensorflow as tf



#``absolute(a - b) <= (atol + rtol * absolute(b))``

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    '''Returns symbolic 'int8' value if all
       elements in tensors a & b are within
       given tolerances'''
    tf_a = tf.cast(tf.constant(a), tf.float32)
    tf_b = tf.cast(tf.constant(b), tf.float32)
    tf_atol = tf.cast(tf.constant(atol), tf.float32)
    tf_rtol = tf.cast(tf.constant(rtol), tf.float32)
    
    tf_ref = tf_atol + tf_rtol * tf.abs(tf_b)
    tf_diff = tf.abs(tf_a - tf_b)

    return tf.reduce_all(tf.less(tf_diff, tf_ref))

def true_div(num, denom):
    return tf.truediv(tf.cast(a,tf.float32), tf.cast(b,tf.float32))

def ECOC_fast_accuracy(y_encode):
    Y_Encoding = tf.constant(y_encode, dtype=K.floatx(), name='Y_Encoding')
    
    def ECOC_fnc(Y_True, Y_Pred):
    
        # Find nearest trgt vector to actual output
        deltas = tf.expand_dims(Y_Pred,1) - tf.expand_dims(Y_Encoding,0) 
        dists = tf.reduce_sum(tf.square(deltas), axis=-1)
        indices = tf.argmin(dists, axis=-1)
        nearest_trgt = tf.squeeze(tf.gather(Y_Encoding, indices)) 
        
        # See if 'best guess' trgt vector is correct
        dists2 = tf.reduce_sum(tf.square(Y_True - nearest_trgt), axis=-1)
        num_right = tf.count_nonzero(dists2 < 1e-06)
        total = tf.size(dists2)
        acc = tf.truediv(tf.cast(num_right, tf.int32), total)
        
        return K.cast(acc, K.floatx())

    return ECOC_fnc
