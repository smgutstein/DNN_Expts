def lr_sched_func(lr, kwargs):

    epoch = kwargs["epoch"] 
    lr_scale = kwargs['lr_scale']
    lr = lr  * (0.5 ** (epoch // lr_scale))

    return lr
