from __future__ import print_function
import hot_not_hot_values as hnh
# import local_backend as LK
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
    


## def make_n_hot_encoding_dict(self, hot=1.0, not_hot=0.0, nb_hot=1):
##     '''Create n-hot codes'''
##     print("Creating %d-hot encodings"%(nb_hot))      
##     # relu hot/not_hot
##     self.not_hot = not_hot
##     self.hot = hot

##     #Make n-hot encoding dict
##     self.encoding_dict = \
##       enc_utils.make_n_hot_nb_2_encoding_dict(self.y_train, self.nb_code_bits,
##                                                  not_hot, hot, nb_hot)
##     #self._record_encoding()
##     self.encoding_type = str(nb_hot) + "-hot"


## def make_n_hot_nb_2_encoding_dict(y, nb_output_nodes, 
##                             not_hot = 0, hot = 1,
##                             nb_hot = 1,
##                             nb_2_encoding_dict = {}):
##     '''Make dict that maps class numbers to encodings'''
    
##     class_nums = list(np.unique(y))
    
##     for ctr,curr_class_num in enumerate(class_nums):
##         #n-hot encoding
##         Y = np.ones((nb_output_nodes)) * not_hot
##         curr_offset = ctr * nb_hot
##         Y[ctr:ctr + nb_hot] = hot
##         nb_2_encoding_dict[curr_class_num] = Y

##     return nb_2_encoding_dict

## '''
## def make_rnd_ecoc_nb_2_encoding_dict(y, nb_output_nodes, 
##                             not_hot = 0, hot = 1,
##                             hot_prob = 0.5,
##                             nb_2_encoding_dict = {}):

##     #Ensure y_train is hashable
##     if len(y.shape) > 2:
##        sys.error("Label data must be stored in 1-D arrays")
##     elif len(y.shape) == 2 and y.shape[1] != 1:
##        sys.error("Label data must be stored in 1-D arrays")
##     elif len(y.shape) == 2:
##        y = y.flatten()
##     else:
##        pass


##     class_nums = list(np.unique(y))
##     code_set = set()

##     # Create number of hot/non-hot values to construct codes with
##     nb_not_hot_nodes = int((1-hot_prob) * nb_output_nodes)
##     nb_hot_nodes = nb_output_nodes - nb_not_hot_nodes
##     sample_set = (hot,)*nb_hot_nodes + (not_hot,)*nb_not_hot_nodes
    
##     for curr_class_num in class_nums:
        
##         #Generate new code
##         Y = tuple(random.sample(sample_set,nb_output_nodes))
##         while Y in code_set:
##            Y = tuple(random.sample(sample_set,nb_output_nodes))

##         code_set.add(Y)
##         nb_2_encoding_dict[curr_class_num] = np.asarray(Y)
        

##     return nb_2_encoding_dict
## '''        
