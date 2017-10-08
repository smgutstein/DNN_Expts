from __future__ import print_function
import hot_not_hot_values as hnh
import numpy as np


def make_encoding_dict(self, **kwargs):
    """ Create n-hot encoding"""
    nb_hot = int(kwargs['nb_hot'])
    hot_func = getattr(hnh, kwargs['hot_not_hot_fnc'])
    hot, not_hot = hot_func(**kwargs)

    print("Creating %d-hot encodings - " % nb_hot,
          "hot/not_hot (%6.4f / %6.4f)" % (hot, not_hot))

    # Get sorted list of class numbers (np.unique returns sorted list)
    class_nums = list(np.unique(self.y_train))
    nb_2_encoding_dict = {}

    for ctr, curr_class_num in enumerate(class_nums):
        # n-hot encoding
        code_word = np.ones(self.nb_code_bits) * not_hot
        code_word[ctr * nb_hot:ctr * nb_hot + nb_hot] = hot
        nb_2_encoding_dict[curr_class_num] = code_word

    self.not_hot = not_hot
    self.hot = hot
    self.encoding_dict = nb_2_encoding_dict
    self.encoding_type = str(nb_hot) + "-hot"

    num_code_words = len(self.encoding_dict)
    sample_code_word = self.encoding_dict.keys()[0]
    code_word_bits = \
    self.encoding_dict[sample_code_word].shape[0]
    self.encoding_matrix = np.zeros((num_code_words,
                                     code_word_bits))
    for ctr, x in enumerate(sorted(self.encoding_dict.keys())):
        self.encoding_matrix[ctr, :] = self.encoding_dict[x]

    self.meta_encoding_dict = dict()
    self.meta_encoding_dict['hot'] = self.hot
    self.meta_encoding_dict['not_hot'] = self.not_hot
    self.meta_encoding_dict['nb_hot'] = nb_hot

