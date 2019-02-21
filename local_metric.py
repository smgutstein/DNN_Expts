from __future__ import absolute_import
from keras import backend as K
import local_backend as LK
import tensorflow as tf

def binary_accuracy_local(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def hot_bit(y_true, y_pred):
    # Equivalent to categorical accuracy function in
    # keras metrics module
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


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

    return TF_ECOC_fnc

'''
Theano version of this function
def TH_ECOC_fast_accuracy(y_encode):
    Y_Encoding = T.constant(y_encode, dtype=K.floatx())

    def TH_ECOC_fnc(Y_True, Y_Pred):
        # Find nearest trgt vector to actual output
        deltas = Y_Pred.reshape((Y_Pred.shape[0], 1, -1)) - Y_Encoding.reshape((1, Y_Encoding.shape[0], -1))
        dists = T.sum(T.pow(deltas, 2), axis=-1)
        nearest_trgt = Y_Encoding[T.argmin(dists, axis=-1)]

        # See if 'best guess' trgt vector is correct
        dists2 = T.sum(T.pow(Y_True - nearest_trgt, 2), axis=-1)
        acc = T.true_div(T.prod(T.shape((dists2 < 1e-06).nonzero())), T.prod(T.shape(dists2)))

        return K.cast(acc, K.floatx())
        
    return ECOC_fnc
'''
