from __future__ import print_function
import argparse
import ConfigParser
import datetime
import errno
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, RadioButtons
import numpy as np
import os
import pickle
import shutil


# Guarantee Theano backend
os.environ["KERAS_BACKEND"] = "theano"

class Weight_Checker(object):

    def __init__(self):
        self.config = ConfigParser.ConfigParser()
        self.cmd_line_args = self.get_cmd_line_args()
        self.set_params()

        self.fig, self.ax = plt.subplots(1)
        self.fig.set_size_inches(9.6, 7.2)

        self.ax.axes.get_xaxis().set_visible(True)
        self.ax.axes.get_yaxis().set_visible(True)
       
        self.axprev = plt.axes([0.1, 0.025, 0.1, 0.05])
        self.axnext = plt.axes([0.20, 0.025, 0.1, 0.05])
        self.axprevlayer = plt.axes([0.75, 0.025, 0.1, 0.05])
        self.axnextlayer = plt.axes([0.85, 0.025, 0.1, 0.05])
        
        self.bnext = Button(self.axnext, 'Next Net')
        self.bnext.on_clicked(self.next)
        
        self.bprev = Button(self.axprev, 'Prev Net')
        self.bprev.on_clicked(self.prev)

        self.bnextlayer = Button(self.axnextlayer, 'Next Layer')
        self.bnextlayer.on_clicked(self.next_layer)
        
        self.bprevlayer = Button(self.axprevlayer, 'Prev Layer')
        self.bprevlayer.on_clicked(self.prev_layer)


    def get_cmd_line_args(self):

        # Get config file from cmd line
        parser = argparse.ArgumentParser(
            description="Run Keras Expt With Specified Output Encoding")
        parser.add_argument('config_files', action='store',
                            type=str, nargs='*', default='')
        parser.add_argument('-i', '--init_iter', action='store',
                            type=int, default=0)
        parser.add_argument('-d', '--delta',  action='store',
                            type=int, default=50)
        parser.add_argument('--gpu', '-g', type=str, default='cuda0',
                            action='store', help='chosen GPU')

        cmd_line_args = parser.parse_args()
        self.iteration = cmd_line_args.init_iter
        self.delta = cmd_line_args.delta

        # Choose specific GPU
        theano_flags = 'mode=FAST_RUN, device=' + cmd_line_args.gpu + ', floatX=32'
        os.environ['THEANO_FLAGS'] = theano_flags
        self.gpu = cmd_line_args.gpu

        global keras
        global K
        global model_from_json, model_from_yaml
        import keras
        from keras import backend as K
        from keras.models import model_from_json, model_from_yaml


        for curr_config_file in (cmd_line_args.config_files):

            # Get input args from config file
            if not os.path.isfile(cmd_line_args.config_files[0]):
                print("Can't find %s. Is it a file?" % cmd_line_args.config_files)
                os._exit(1)

            yield curr_config_file

    def get_param_dict(self, dict_name):

        param_dict = {}
        try:
            params = self.config.items(dict_name)
            for curr_pair in params:
                param_dict[curr_pair[0]] = curr_pair[1]
        except ConfigParser.NoSectionError:
            pass
        return param_dict


    def set_params(self):
        try:
            self.expt_file_name = self.cmd_line_args.next()
        except StopIteration as exception:
            return False

        self.config.read(self.expt_file_name)
        self.saved_param_dict = self.get_param_dict('SavedParams')
        self.src_dir = os.path.join(self.saved_param_dict['saved_set_dir'],
                                    self.saved_param_dict['saved_dir'])

        with open(os.path.join(self.src_dir,
                               self.saved_param_dict['saved_dir'] + '_init.' +
                               self.saved_param_dict['saved_arch_format']), 'r') as f:

            if self.saved_param_dict['saved_arch_format'][-4:] == 'json':
                json_str = f.read()
                self.model = model_from_json(json_str)
            elif self.saved_param_dict['saved_arch_format'][-4:] == 'yaml':
                yaml_str = f.read()
                self.model = model_from_yaml(yaml_str)

    def get_weights_filename(self):
        wt_file = os.path.join(self.src_dir,
                               self.saved_param_dict['saved_dir'] +
                               '_weights_' +
                               str(self.iteration) + '.h5')
        return wt_file


    def get_weights(self):

        wt_file = self.get_weights_filename()
        if os.path.isfile(wt_file):
            self.wt_file = wt_file
            self.model.load_weights(self.wt_file)

            self.weight_dict = {}
            ctr = 0
            for curr_layer in self.model.layers:
                curr_weight_values = curr_layer.get_weights()
                curr_weight_names = curr_layer.weights
                if curr_weight_names:
                    for cwn,cwv in zip(curr_weight_names,
                                       curr_weight_values):
                        w_array = cwv.flatten()
                        num_bins = round(1 +
                                         3.22 * 
                                         math.log(len(w_array),10)) #Sturges Rule
                        lims = [min(w_array), max(w_array)]
                        if abs(lims[0]-lims[1]) < 0.05:
                            lims[0] -= 0.05
                            lims[1] += 0.05
                        lims_20 = [math.floor(x*20)/20. for x in lims]
                        bins = np.linspace(lims_20[0], lims_20[1], num_bins)
                        self.weight_dict[ctr] = (cwn.name, w_array, bins)
                        ctr += 1
                    
            success = True
        else:
            success = False

        return success

    def get_curr_title(self):
        return "Layer : " + self.weight_dict[self.curr_layer][0] + "\nEpoch: " + str(self.iteration)


    def next(self, event):
        self.iteration += self.delta
        success = self.get_weights()
        if success:
            self.refresh_image()
            self.ax.set_title(self.get_curr_title())
            #plt.title(str(self.iteration), loc='left')
            event.canvas.draw()
        else:
            print ("Unable to load {}".format(self.get_weights_filename))
            self.iteration -= self.delta
        
            
    def prev(self, event):
        self.iteration -= self.delta
        success = self.get_weights()
        if success:
            self.refresh_image()
            self.ax.set_title(self.get_curr_title())
            #plt.title(str(self.iteration), loc='left')
            event.canvas.draw()
        else:
            print ("Unable to load {}".format(self.get_weights_filename))
            self.iteration += self.delta

    
    def next_layer(self, event):
        self.curr_layer += 1
        if self.curr_layer > max(self.weight_dict.keys()):
            self.curr_layer = min(self.weight_dict.keys())

        self.refresh_image()
        self.ax.set_title(self.get_curr_title())
        event.canvas.draw()
        
    def prev_layer(self, event):
        self.curr_layer -= 1
        if self.curr_layer < min(self.weight_dict.keys()):
            self.curr_layer = max(self.weight_dict.keys())

        self.refresh_image()
        self.ax.set_title(self.get_curr_title())
        event.canvas.draw()
        
            
    def start_display(self):
        self.get_weights()
        self.curr_layer = 0
        self.refresh_image()
        #plt.ion()
        #plt.draw()
        self.ax.set_title(self.get_curr_title())
        plt.show()

    def refresh_image(self):
        self.ax.clear()
        curr_data = self.weight_dict[self.curr_layer][1]
        bins = self.weight_dict[self.curr_layer][2]
        self.ax.hist(curr_data, bins=bins)
        #self.fig.canvas.draw()
        #plt.show()
        #self.fig.show()
 
if __name__ == '__main__':

    x = Weight_Checker()
    x.start_display()
