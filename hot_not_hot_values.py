from keras import activations
import scipy
from theano import tensor as T
from theano import function as Tfunc


def hot_not_hot_2nd_deriv_max(**kwargs):
    x = T.dscalar()
    f = activations.get(kwargs['encoding_activation_fnc'])
    y = f(x)
    d2y = T.grad(T.grad(y, x), x)
    Y2 = Tfunc([x], d2y)
    mid = scipy.optimize.brentq(Y2, -10, 10)
    d3y = T.grad(d2y, x)
    Y3 = Tfunc([x], d3y)
    hot_x = scipy.optimize.brentq(Y3, mid, 10)
    not_hot_x = scipy.optimize.brentq(Y3, -10, mid)
    Y = Tfunc([x], y)
    hot = float(Y(hot_x))
    not_hot = float(Y(not_hot_x))

    return hot, not_hot

def hot_not_hot_softmax(**kwargs):
    hot = float(kwargs['hot_val'])
    not_hot = (1. - hot) / (float(kwargs['nb_code_bits']) - 1)

    return hot, not_hot

def hot_not_hot_1_0(**kwargs):
    hot = 1.0
    not_hot = 0.0

    return hot, not_hot

def hot_not_hot_1_neg1(**kwargs):
    hot = 1.0
    not_hot = -1.0

    return hot, not_hot

def hot_not_hot_handset(**kwargs):
    hot = float(kwargs['hot'])
    not_hot = float(kwargs['not_hot'])

    return hot, not_hot
