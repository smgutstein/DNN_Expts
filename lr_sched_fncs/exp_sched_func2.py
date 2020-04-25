def lr_sched_func(lr, kwargs):

    curr_epoch = kwargs["epoch"] 
    lr_decay = kwargs['lr_decay']
    start_epoch = kwargs['start_epoch']
    stop_epoch = kwargs['stop_epoch']
    
    if (curr_epoch >= start_epoch) and (curr_epoch <= stop_epoch):
        lr = lr_decay*lr

    return lr
