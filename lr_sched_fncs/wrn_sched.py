def lr_sched_func(lr, lr_orig, kwargs):

    epoch = kwargs["epoch"]
    drop = 0.2

    if epoch < 60:
        pwr = 0
    elif epoch < 120 and epoch >= 60:
        pwr = 1
    elif epoch < 160 and epoch >= 120:
        pwr = 2
    else:
        pwr = 3
        
    lr = lr_orig * drop**(pwr)
    print("lr = ", lr)

    return lr
