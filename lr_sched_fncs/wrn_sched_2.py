def lr_sched_func(lr, lr_orig, kwargs):

    epoch = kwargs["epoch"]
    drop = 0.2

    if epoch < 100:
        pwr = 0
    elif epoch < 160 and epoch >= 100:
        pwr = 1
    elif epoch < 180 and epoch >= 160:
        pwr = 2
    else:
        pwr = 3
        
    lr = lr_orig * drop**(pwr)
    print("lr = ", lr)

    return lr
