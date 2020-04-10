def lr_sched_func(kwargs):

    curr_batch = kwargs["batch"] 
    lr_decay = kwargs['lr_decay']
    lr_decay_steps = kwargs['lr_decay_steps']
    initial_lr = kwargs['base_lr']
    warmup_epochs = kwargs['warmup_epochs']
    steps_per_epoch = kwargs['steps_per_epoch']

    decay_factor = lr_decay * (curr_batch / lr_decay_steps)
    lr = initial_lr * decay_factor
    
    num_warmup_batches = steps_per_epoch * warmup_epochs
    if (warmup_epochs > 0) and (curr_batch < num_warmup_batches):

        # Create node with value of learning rate by multiplying a number
        # by a node, and then dividing by a number
        lr = initial_lr  * curr_batch / num_warmup_batches

    return lr

on_batch=True
on_epoch=False
