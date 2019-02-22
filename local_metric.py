from __future__ import absolute_import
from keras import backend as K
import local_backend as LK
from local_backend import ECOC_fast_accuracy

def binary_accuracy_local(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def hot_bit(y_true, y_pred):
    # Equivalent to categorical accuracy function in
    # keras metrics module
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())




