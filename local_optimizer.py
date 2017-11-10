from keras import optimizers

def sgd(sgd_params):
    # Stochastic gradient descent
    # e.g. SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    return optimizers.SGD(**sgd_params)