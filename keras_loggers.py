from __future__ import print_function

from keras.callbacks import BaseLogger, Callback
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import pickle
import shutil


class ModelCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor, verbose=0,
                 save_best=True, save_most_recent=True, save_weights_only=False,
                 mode='auto', period=1, data_manager=None, nocheckpoint=False):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best = save_best
        self.save_most_recent = save_most_recent
        self.save_weights_only = save_weights_only
        self.period = period
        self.data_manager = data_manager
        self.epochs_since_last_save = 0
        self.best_epoch = None
        self.encodings_saved = False
        self.nocheckpoint = nocheckpoint

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best_score = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best_score = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best_score = -np.Inf
            else:
                self.monitor_op = np.less
                self.best_score = np.Inf

    def save_best_model(self, epoch, logs={}):

        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Can save best model only with %s available, '
                          'skipping.' % (self.monitor), RuntimeWarning)
        else:
            if self.monitor_op(current, self.best_score):
                outfile = self.filepath + "_best_weights_" + str(epoch) + ".h5"
                if self.verbose > 0:
                    print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                          ' saving model to %s'
                          % (epoch, self.monitor, self.best_score,
                             current, outfile))

                # Remove prior best model
                curr_best_file = self.filepath + \
                                 "_best_weights_" + str(self.best_epoch)
                curr_data_manager_file = self.filepath + \
                                         '_encodings_' + \
                                         str(self.best_epoch) + '.pkl'
                if os.path.isfile(curr_best_file + '.h5'):
                    os.remove(curr_best_file + '.h5')

                # Save new best encodings
                # Note: Don't remove old "best encodings" file, since it is used
                # for saved epochs other than best
                if not self.encodings_saved:
                    self.save_data_manager(curr_data_manager_file)

                # Save new best model
                self.best_score = current
                self.best_epoch = epoch
                if self.nocheckpoint:
                    return
                elif self.save_weights_only:
                    self.model.save_weights(outfile, overwrite=True)
                else:
                    self.model.save(outfile, overwrite=True)

            else:
                if self.verbose > 0:
                    print('Epoch %05d: %s did not improve from %0.5f' %
                          (epoch, self.monitor, self.best_score))


    def save_data_manager(self, trgt_file):

        self.data_manager.curr_encoding_info['encoding_dict'] = self.data_manager.encoding_dict
        self.data_manager.curr_encoding_info['label_dict'] = self.data_manager.label_dict
        self.data_manager.curr_encoding_info['meta_encoding_dict'] = self.data_manager.meta_encoding_dict

        print ("Saving", trgt_file)
        with open(trgt_file, 'wb') as f:
            pickle.dump(self.data_manager.curr_encoding_info, f)
        self.encodings_saved = True


    def save_net(self, trgt_file):

        if self.nocheckpoint:
            return
        
        if self.save_weights_only:
            self.model.save_weights(trgt_file, overwrite=True)
        else:
            self.model.save(trgt_file, overwrite=True)
            

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        logs = logs or {}
 
        curr_weights_file = self.filepath + '_weights_' + str(epoch) + '.h5'
        old_weights_file = self.filepath + '_weights_' + str(epoch - 1) + '.h5'
        curr_data_manager_file = self.filepath + '_encodings_' + str(epoch) + '.pkl'
        #old_data_manager_file = self.filepath + '_weights_' + str(epoch - 1) + '.pkl'


        if (epoch % self.period) == 0:
            # Save every nth epoch
            self.save_net(curr_weights_file)
            #self.save_data_manager(epoch, "weights")
        elif self.save_most_recent:
            # Save most recent epoch
            self.save_net(curr_weights_file)
            #self.save_data_manager(epoch, "weights")

        if not self.encodings_saved:
            self.save_data_manager(curr_data_manager_file)

        if ((epoch-1) % self.period) != 0:
            # Delete last "most recent" epoch, if its doesnt
            # interfere with saving every nth epoch
            if os.path.isfile(old_weights_file):
                os.remove(old_weights_file)
            #if os.path.isfile(old_data_manager_file):
            #    os.remove(old_data_manager_file)


        # Check to see if copy to best_epoch and delete last best epoch
        if self.save_best:
            self.save_best_model(epoch, logs)

        print("\n========================================================================\n")

                    

# Monitor taken from PyImageSearch

class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, resultsPath=None, startAt=0):
        # store the output path for the figure, the path to the JSON
        # serialized file, and the starting epoch
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.resultsPath = resultsPath
        self.startAt = startAt
                
        self.acc_fig = plt.figure(1)
        plt.clf()
        self.acc_ax = self.acc_fig.add_axes([0.15, 0.1, 0.8, 0.8])
                
        self.loss_fig = plt.figure(2)
        plt.clf()
        self.loss_ax = self.loss_fig.add_axes([0.15, 0.1, 0.8, 0.8])
                

    def on_train_begin(self, logs={}):
        if not hasattr(self, 'H'):
            # initialize the history dictionary
            self.H = {}

        # if the JSON history path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                # check to see if a starting epoch was supplied
                if self.startAt > 0:
                    # loop over the entries in the history log and
                    # trim any entries that are past the starting
                    # epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        # check to see if the training history should be serialized
        # to file
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()

            # make human readable text file
            if self.resultsPath is not None:
                epoch_num = len(self.H["loss"]) - 1
                out_str = 'Epoch ' + str(epoch_num) + ': '
                out_fields = sorted(self.H)
                for curr_out in out_fields:
                    out_str += '{}: {:5.4f}  '.format(curr_out, self.H[curr_out][-1])
                with open(self.resultsPath, 'a') as f:
                    f.write(out_str + '\n')

        # ensure at least two epochs have passed before plotting
        # (epoch starts at zero)
        if len(self.H["loss"]) > 1:
            plt.style.use("ggplot")
            self.acc_ax.set_title("Accuracy vs. Epochs [{}]".format(len(self.H["loss"])))
            self.loss_ax.set_title("Loss vs. Epochs [{}]".format(len(self.H["loss"])))
            N = np.arange(0, len(self.H["loss"]))
            num_epochs = len(self.H["loss"]) - 1
                    
            plt.figure(1) # acc                   
            plt.cla()
            self.acc_ax.set_xlabel("Epochs")
            self.acc_ax.set_ylabel("Acc")

            tr_accs = sorted([x for x in self.H if 'val' not in x and 'acc' in x])
            title_str = "Accuracy [Epoch {} | Acc ({:5.2f}%".format(num_epochs,
                                                                    100*max(self.H[tr_accs[0]]))
            plt.plot(N, self.H[tr_accs[0]], label=tr_accs[0])
                    
            if len(tr_accs) > 1:
                for curr_plt in tr_accs:
                    plt.plot(N, self.H[curr_plt], label=curr_plt)
                    title_str += ", {:5.2f}%".format(100*max(self.H[curr_plt]))

            val_accs = sorted([x for x in self.H if 'val' in x and 'acc' in x])
            for curr_plt in val_accs:
                plt.plot(N, self.H[curr_plt], label=curr_plt)
                title_str += ", {:5.2f}%".format(100*max(self.H[curr_plt]))
            title_str += ") ]"
            plt.title(title_str)


            plt.legend()
            plt.savefig(self.figPath[0])

            plt.figure(2) # loss
            plt.cla()
            self.loss_ax.set_xlabel("Epochs")
            self.loss_ax.set_ylabel("Loss")               
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.title("Loss [Epoch {}]".format(num_epochs))
            plt.legend()
            plt.savefig(self.figPath[1])

