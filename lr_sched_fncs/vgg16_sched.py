def lr_sched_func(lr, lr_orig, kwargs):

    epoch = kwargs["epoch"] 
    lr_scale = kwargs['lr_scale']

    lr = lr_orig  * (0.5 ** (epoch // lr_scale))
    print("lr = ", lr, " * ", "0.5 ** (", epoch, "//", lr_scale,")")

    return lr
