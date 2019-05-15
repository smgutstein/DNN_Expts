from __future__ import absolute_import
from keras import backend as K
import local_backend as LK
from local_backend import ECOC_fast_accuracy
from local_backend import ECOC_top_1, ECOC_top_5

def binary_accuracy_local(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def hot_bit_wrapper():
    def hot_bit(y_true, y_pred):
        # Equivalent to categorical accuracy function in
        # keras metrics module
    
        return K.cast(K.equal(K.argmax(y_true, axis=-1),
                              K.argmax(y_pred, axis=-1)),
                      K.floatx())
    hot_bit.__name__ = "acc_top_1"
    return hot_bit

def top_k_categorical_accuracy_wrapper(k):
    def top_k_categorical_accuracy(y_true, y_pred, k=5):
        return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)
    top_k_categorical_accuracy.__name__ = "acc_top_" + str(k)
    return top_k_categorical_accuracy
    
# Slightly kludgy way of changing function name, but
# maintaining backwards compatibility
hot_bit = hot_bit_wrapper()

hot_bit_top_5 = top_k_categorical_accuracy_wrapper(5)
