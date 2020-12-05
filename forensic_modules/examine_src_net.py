from __future__ import print_function
import argparse
import configparser
from keras.models import model_from_json, model_from_yaml
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import os
import sys


class WeightChecker(object):

    def __init__(self):
        self.config = configparser.ConfigParser()
        self.forensics_file = self.get_cmd_line_args()
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

    @staticmethod
    def get_cmd_line_args():

        # Get config file from cmd line
        parser = argparse.ArgumentParser(
            description="Run Keras Expt With Specified Output Encoding")
        parser.add_argument('forensics_file', action='store',
                            type=str, nargs='*', default='')

        cmd_line_args = parser.parse_args()
        forensics_file = os.path.join('./cfg_dir/net_forensics_cfg',
                                      cmd_line_args.forensics_file[0])

        # Get input args from config file
        if not os.path.isfile(forensics_file):
            print("Can't find %s. Is it a file?" % forensics_file)
            sys.exit()

        return forensics_file

    def get_param_dict(self, dict_name):

        param_dict = {}
        try:
            params = self.config.items(dict_name)
            for curr_pair in params:
                param_dict[curr_pair[0]] = curr_pair[1]
        except configparser.NoSectionError:
            pass
        return param_dict

    def set_params(self):

        self.config.read(self.forensics_file)
        self.forensics_file_param_dict = self.get_param_dict('PathParams')

        self.expt_cfg_file = os.path.join(self.forensics_file_param_dict['root_dir'],
                                          self.forensics_file_param_dict['expt_dir'],
                                          self.forensics_file_param_dict['arch_dir'],
                                          self.forensics_file_param_dict['net_type'],
                                          self.forensics_file_param_dict['cfg_file'])
        if not os.path.isfile(self.expt_cfg_file):
            print("Can't find %s. Is it a file?" % self.expt_cfg_file)
            sys.exit()

        self.config.read(self.expt_cfg_file)
        self.expt_file_param_dict = self.get_param_dict('ExptFiles')
        self.net_param_dict = self.get_param_dict('NetParams')
        self.results_root_dir = os.path.join(self.expt_file_param_dict['root_expt_dir'],
                                             self.expt_file_param_dict['expt_dir'],
                                             self.expt_file_param_dict['expt_subdir'],
                                             self.forensics_file_param_dict['machine_name'])

        init_arch = [x for x in os.listdir(os.path.join(self.results_root_dir,
                                                        'checkpoints'))
                     if 'arch' in x][0]
        with open(os.path.join(self.results_root_dir,
                               'checkpoints',
                               init_arch), 'r') as f:

            if init_arch[-4:] == 'json':
                json_str = f.read()
                self.model = model_from_json(json_str)
            elif init_arch[-4:] == 'yaml':
                yaml_str = f.read()
                self.model = model_from_yaml(yaml_str)

        def net_sorter(x):
            return int(x.split('_')[-1].split('.')[0])

        self.net_list = [x for x in os.listdir(os.path.join(self.results_root_dir,
                                               'checkpoints'))
                         if 'weight' in x]
        self.net_list.sort(key=net_sorter)
        self.curr_net = 0

    def get_weights_filename(self):
        wt_file = os.path.join(self.results_root_dir,
                               'checkpoints',
                               self.net_list[self.curr_net])
        return wt_file

    def get_epoch(self):
        return int(self.net_list[self.curr_net].split('_')[-1].split('.')[0])

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
                                         math.log(len(w_array), 10))  # Sturges Rule
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
        return "Layer : " + self.weight_dict[self.curr_layer][0] + "\nEpoch: " + str(self.get_epoch())

    def next(self, event):
        self.curr_net += 1
        if self.curr_net >= len(self.net_list):
            self.curr_net = len(self.net_list) - 1
        success = self.get_weights()
        if success:
            self.refresh_image()
            self.ax.set_title(self.get_curr_title())
            event.canvas.draw()
        else:
            print ("Unable to load {}".format(self.get_weights_filename))
            self.curr_net -= 1

    def prev(self, event):
        self.curr_net -= 1
        self.curr_net = max(0, self.curr_net)
        success = self.get_weights()
        if success:
            self.refresh_image()
            self.ax.set_title(self.get_curr_title())
            event.canvas.draw()
        else:
            print ("Unable to load {}".format(self.get_weights_filename))
            self.curr_net += 1

    
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
        self.ax.set_title(self.get_curr_title())
        plt.show()

    def refresh_image(self):
        self.ax.clear()
        curr_data = self.weight_dict[self.curr_layer][1]
        bins = self.weight_dict[self.curr_layer][2]
        self.ax.hist(curr_data, bins=bins)
 

if __name__ == '__main__':

    x = WeightChecker()
    x.start_display()
