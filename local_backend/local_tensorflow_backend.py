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


def ECOC_top_k_accuracy(top_k):    

    def ECOC_top_k_wrapper(y_encode):
        Y_Encoding = tf.constant(y_encode,
                                 dtype=K.floatx(), name='Y_Encoding')
    
        def ECOC_top_k_fnc(Y_True, Y_Pred):

            # Repeat each row of Y_True top_k times, consecutively
            temp_expand = tf.expand_dims(Y_True,-1)
            temp_tile = tf.tile(temp_expand, [1, top_k,1])
            Y_True_top_k = tf.reshape(temp_tile,[tf.shape(Y_True)[0]*top_k,
                                                 tf.shape(Y_True)[1]])

            # Get distance betwen each output and each 'possible' trgt vector 
            # the output could be approximating
            deltas = tf.expand_dims(Y_Pred,1) - tf.expand_dims(Y_Encoding,0) 
            dists = tf.reduce_sum(tf.square(deltas), axis=-1)

            # Get top_k closest 'possible' trgt vector indices for each output
            _, temp_k_indices = tf.nn.top_k(tf.negative(dists), k=top_k)
            top_k_indices = tf.reshape(temp_k_indices,[-1])

            # Get top_k closest 'possible' trgt vector for each output
            temp_trgt_k = tf.squeeze(tf.gather(Y_Encoding,top_k_indices))
            nearest_trgt_top_k = tf.reshape(temp_trgt_k,
                                            [tf.shape(Y_True_top_k)[0],
                                             tf.shape(Y_True_top_k)[1]])

            # Get top_k closest 'possible' distances to trgt encodings
            dists2_top_k = tf.reduce_sum(tf.square(Y_True_top_k - nearest_trgt_top_k), axis=-1)

            # Get accuuracy (num_right/ total). Since tot has top_k times more entries
            # than it should (due to repetition in Y_True_top_k), num_right is multiplied
            # by top_k
            num_right_top_k = tf.count_nonzero(dists2_top_k  < 1e-06)*top_k
            total_top_k = tf.count_nonzero(dists2_top_k  > -1)
            acc = tf.truediv(num_right_top_k, total_top_k)  

            return K.cast(acc, K.floatx())
        ECOC_top_k_fnc.__name__ = "acc_top_" + str(top_k)
        return ECOC_top_k_fnc
    
    return ECOC_top_k_wrapper

ECOC_top_5 = ECOC_top_k_accuracy(5)
ECOC_top_1 = ECOC_top_k_accuracy(1)
