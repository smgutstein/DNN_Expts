import os
import pickle
import tensorflow.keras as keras


class PrunerCallback(keras.callbacks.Callback):
    """  """
    def __init__(self, pruner, use_dwr=False):
        """ A keras callback that prunes weights using a `LotteryTicketPruner`.
        Per the intention of lottery ticket pruning the model being trained 
        is pruned at the beginning of every epoch so
        that training is done with the pruned weights set to zero.
        After completion of training the model will also be pruned so that 
        the final trained model has pruning applied for inference.
        :param pruner: A `LotteryTicketPruner` instance that is used to prune 
         weights during and just after training. The pruner is only used to apply
         the pruning mask to the model's weights. The caller should make sure that
         this pruner instance's `LotteryTicketPruner.prune_weights()` has been 
         called to calculate the pruning mask.
        :param use_dwr: If True then the callback will apply 
         Dynamic Weight Rescaling (DWR) to the unpruned weights in the model 
         after every epoch.
         See section 5.2, "Dynamic Weight Rescaling" of
         https://arxiv.org/pdf/1905.01067.pdf.
         A quote from that paper describes it best:
         "For each training iteration and for each layer, we multiply 
          the underlying weights by the ratio of the total number of weights 
          in the layer over the number of ones in the corresponding mask."
        """
        super().__init__()
        self.pruner = pruner
        self.use_dwr = use_dwr

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        # Prune weights after training is completed so inference
        # uses pruned weights
        self.pruner.apply_pruning(self.model)
        # Save mask
        cpr = int(self.pruner.cumulative_pruning_rate * 10000)
        mask_name = "_".join(["cpr", str(cpr)]) + ".pkl"
        mask_path = os.path.join(self.pruner.saved_mask_dir, mask_name)
        #  Ensure self.pruner.saved_mask_dir exists
        if self.pruner.saved_mask_dir is not None:
            try:
                os.makedirs(self.pruner.saved_mask_dir)
            except OSError:
                if not os.path.isdir(self.pruner.saved_mask_dir):
                    raise

        pickle.dump(self.pruner.prune_masks_map, open(mask_path, 'wb'))
        print ("Saved cumulative prune rate map:{} to {}".format(cpr, self.pruner.saved_mask_dir))
        # Don't apply DWR at the end of training since it changes
        # the weights that we just trained so hard to arrive at

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        # End of epoch so prune the weights that we're pruning
        self.pruner.apply_pruning(self.model)
        perc_pruned = self.pruner.pruned_weights / self.pruner.tot_weights
        print("{:5.4f} percent of {:.2e} parameters pruned".format(perc_pruned,
                                                                   self.pruner.tot_weights))
        if self.use_dwr:
            self.pruner.apply_dwr(self.model)

    def on_batch_end(self, batch, logs=None):
        super().on_train_end(logs)
        # Prune weights after each batch to ensure
        # pruned weights don't get trained
        self.pruner.apply_pruning(self.model)
        pass


