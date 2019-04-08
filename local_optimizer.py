from keras import optimizers

def sgd(sgd_params):
    # Stochastic gradient descent
    # e.g. SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    return optimizers.SGD(**sgd_params)


def adam(adam_params):
    # ADAM - ADAptive Moment estimation
    # e.g. Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
    #             epsilon=None, decay=0., amsgrad=False)
    return optimizers.Adam(**adam_params)
