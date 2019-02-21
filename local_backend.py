from collections import defaultdict
from contextlib import contextmanager
import scipy
import numpy as np
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

