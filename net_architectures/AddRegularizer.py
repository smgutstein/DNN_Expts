# Copyright 2019 The TensorFlow Authors, Pavel Yakubovskiy, Björn Barz. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for EfficientNet model.

[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""

# Code of this model implementation is mostly written by
# Björn Barz ([@Callidior](https://github.com/Callidior))

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import json
import keras
import math
from net_architectures.NetMapper import map_network
import numpy as np
import os
from six.moves import xrange
import string

####################################################

def Add_Regularizers(inputs, x, regularizer)
    # Map out net and identify layers to which regularizer
    # will be added
    layer_dict = map_network(inputs, x)
    regularizer_layers = dict()
    for k,v in layer_dict.items():
        reg_layer_list = [x for x in v
                          if (hasattr(x,'kernel_regularizer') or
                              hasattr(x,'bias_regularizer'))]
        if reg_layer_list != []:
           regularizer_layers[k] = reg_layer_list

    for curr_layer in regularizer_layers:
        for curr_node in curr_layer:
            if 'kernel_regularizer' in curr_node.__dict__:
                curr_node.kernel_initializer = regularizer
            if 'bias_regularizer' in curr_node.__dict__:
                curr_node.bias_initializer = regularizer
            if 'depthwise_regularizer' in curr_node.__dict__:
                curr_node.depthwise_initializer = regularizer
