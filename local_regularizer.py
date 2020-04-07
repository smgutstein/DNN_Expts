from __future__ import print_function

from keras import backend as K
from keras import regularizers
from keras.regularizers import Regularizer, L1L2
from keras.legacy import interfaces
import sys
import tensorflow as tf

# Note: Real purpose of this file is dedicated location for net_manager
#       to find user generated regularizers. This might be overkill.

# Aliases.


def l1(l1=0.01):
    return L1L2(l1=l1)


def l2(l2=0.01):
    return L1L2(l2=l2)


def l1_l2(l1=0.01, l2=0.01):
    return L1L2(l1=l1, l2=l2)
