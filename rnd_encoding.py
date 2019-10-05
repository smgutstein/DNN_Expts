from __future__ import print_function
from datetime import datetime
import hot_not_hot_values as hnh
from random import random
import numpy as np


def make_encoding_dict(self, **kwargs):
    """ Create rnd encoding"""
    hot_func = getattr(hnh, kwargs['hot_not_hot_fnc'])
    hot, not_hot = hot_func(**kwargs)
    hot_prob = float(kwargs['hot_prob'])
    if 'seed' in kwargs and len(kwargs['seed']) > 0:
        seed = float(kwargs['seed'])
    else:
        seed = datetime.now().microsecond


    print("Creating random encodings:")
    print("   Prob hot: {:6.4f}".format(hot_prob))
    print("   Hot/Not_Hot {:6.4f} / {:6.4f}".format(hot, not_hot))
    print("   Seed: ", seed)

    self.encoding_matrix = \
        np.array([ [ hot if random() <= 0.5 else not_hot
                     for x in range(self.nb_code_bits)]
                   for y in range(len(self.class_nums))])

    self.encoding_dict = {}

    for ctr, curr_class_num in enumerate(self.class_nums):
        self.encoding_dict[curr_class_num] = self.encoding_matrix[ctr, :]

    self.not_hot = not_hot
    self.hot = hot
    self.encoding_type = "Random({:6.4f}".format(hot_prob)

    self.meta_encoding_dict = dict()
    self.meta_encoding_dict['hot'] = self.hot
    self.meta_encoding_dict['not_hot'] = self.not_hot
    self.meta_encoding_dict['encoding_type'] = self.encoding_type
    self.meta_encoding_dict['seed'] = seed
    self.meta_encoding_dict['hot_prob'] = hot_prob



