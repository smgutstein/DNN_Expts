from __future__ import absolute_import
from keras import backend as K
import local_backend as LK
from keras.backend import _BACKEND
from theano import tensor as T
from theano import scan as theano_scan
from theano import function as Tfunc
from theano import In as SymbolicInput
    

def binary_accuracy_local(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def hot_bit(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


def nearest_target(x, targets):
    #Generate Cartesian Product of rows to compare
    i = T.ivector('i')
    j = T.ivector('j')

    i_values = T.arange(x.shape[0]).repeat(targets.shape[0]) #outer rows/loop
    j_values = T.tile(T.arange(targets.shape[0]), x.shape[0]) #inner rows/loop

    #Get dists between each pair of rows in x & targets
    dists,updates = theano_scan(lambda ii, jj, xx, yy: T.sum((xx[ii,:] - yy[jj,:])**2), 
                              sequences=[i_values ,j_values], non_sequences=[x, targets])
    
    #Get indices of rows in targets that are closest to each row of x
    indices = T.flatten(T.argmin(dists.reshape((x.shape[0], targets.shape[0])), axis = -1))

    #Recover each targets-row from the indices
    nn, updates = theano_scan(lambda idx, targets: targets[idx,:],
                              sequences=[indices],
                              non_sequences=targets) 

    return nn

def ECOC_accuracy(y_encode):
    def calc_acc(y_true, y_pred):
        acc_list, updates = theano_scan(fn=lambda a,b:LK.allclose(a,b),
                               sequences=[y_true, y_pred])
        return LK.true_div(acc_list.sum(), acc_list.size)
    
    def ECOC_fnc(y_true, y_pred):
        return K.cast(K.mean(calc_acc(y_true,
                      nearest_target(y_pred,y_encode))),
                      K.floatx())
    return ECOC_fnc


