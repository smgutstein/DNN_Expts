def lr_sched_func(lr, kwargs):

    curr_epoch = kwargs["epoch"] 
    lr_decay = kwargs['lr_decay']
    warmup_epochs = kwargs['warmup_epochs']
    
    if (curr_epoch >= warmup_epochs):
        lr = lr_decay*lr

    return lr
