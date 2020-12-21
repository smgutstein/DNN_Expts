import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict, defaultdict
import keras
from keras.models import model_from_json, model_from_yaml
import numpy as np
import os
import pickle
import sys

def is_number(in_str):
    try:
        float(in_str)
        return True
    except ValueError:
        return False


class NetWeights(object):

    def __init__(self, results_dir, pkl_file = "mag_dict.pkl",
                 pdf_file = "graphs.pdf", net_id='best_key'):
        self.results_root_dir = results_dir
        self.net_id = net_id
        self.pkl_file = pkl_file
        self.pdf_file = pdf_file
        self.epoch_mag_dict = None
        self.init_nets()

    def init_nets(self):
        '''Gets net architecture & stores list of training snapshot names'''
        # Find file specifying net architecture
        init_arch = [x for x in os.listdir(os.path.join(self.results_root_dir,
                                                        'checkpoints'))
                     if 'arch' in x][0]

        # Load net architecture
        arch_file_name = os.path.join(self.results_root_dir,
                                      'checkpoints',
                                      init_arch)
        with open(arch_file_name, 'r') as f:

            print("Loading Architecture from: ", arch_file_name)
            if init_arch[-4:] == 'json':
                json_str = f.read()
                self.model = model_from_json(json_str)
            elif init_arch[-4:] == 'yaml':
                yaml_str = f.read()
                self.model = model_from_yaml(yaml_str)

        # Get list of saved nets and sort by epochs trained
        def net_id(x):
            return int(x.split('_')[-1].split('.')[0])

        net_list = [x for x in os.listdir(os.path.join(self.results_root_dir,
                                               'checkpoints'))
                         if 'weight' in x]
        net_list.sort(key=net_id)
        self.net_dict = OrderedDict()
        for curr in net_list:
            self.net_dict[net_id(curr)] = curr
            if 'best' in curr:
                temp = net_id(curr)
                self.best_epoch = net_id(curr)
        self.net_dict['best_key'] = temp
        
    def get_net(self, net_id):
        # Load weights for specified net & rebuild it. net_id indicates either
        # the number of training epochs used to create the net, or that this
        # net had the best testing set performance
        self.curr_epoch = self.net_dict[net_id] if net_id == 'best_key' else net_id
        self.get_weights()
        
    def get_weights(self):
        # Load saved weights into model
        wt_file = os.path.join(self.results_root_dir,
                               'checkpoints',
                               self.net_dict[self.curr_epoch])

        if os.path.isfile(wt_file):
            print("Loading weights from: ", wt_file)
            self.model.load_weights(wt_file)
        else:
            print("Could not find %s. Is it a file ?\n"% wt_file)
            sys.exit()
            
    def get_trainable_layers(self):
        self.trainable_layers = [x 
                            for x in self.model.layers 
                            if 'conv' in x.name or 'dense' in x.name]
        
        self.net_info = {}
        for curr_layer in self.trainable_layers:
            if 'conv' in curr_layer.name:
                weight_array = curr_layer.get_weights()[0]
                num_maps = weight_array.shape[-1]
                vec_dim = weight_array.shape[:-1]
                self.net_info[curr_layer.name] = [vec_dim, num_maps]
            elif 'dense' in curr_layer.name:
                vec_dim = curr_layer.get_weights()[0].flatten().shape
                self.net_info[curr_layer.name] = [vec_dim, 1]
        return None
    
    def make_weight_dict(self):
        wd = OrderedDict()
        for curr_layer in self.trainable_layers:
            if 'conv' in curr_layer.name:
                weight_array = curr_layer.get_weights()[0]
                num_maps = weight_array.shape[-1]
                map_dict = {curr_map:weight_array[:,:,:,curr_map].flatten()
                            for curr_map in range(num_maps)}
                wd[curr_layer.name] = map_dict


                if len(curr_layer.get_weights()) != 1:
                    print("Error - extra conv weights")
                    sys.exit()
            elif 'dense' in curr_layer.name:
                wd[curr_layer.name] = curr_layer.get_weights()[0].flatten()
        return wd
    
    def get_epoch_mags(self, in_dict):
        mag_dict = OrderedDict()
        for curr_layer in in_dict:
            if 'conv' in curr_layer:
                map_dict = in_dict[curr_layer]
                magnitude_dict = {curr_map:np.linalg.norm(map_dict[curr_map])
                                  for curr_map in map_dict}
                mag_dict[curr_layer] = magnitude_dict
            elif 'dense' in curr_layer:
                mag_dict[curr_layer] = np.linalg.norm(in_dict[curr_layer])
        return mag_dict   
    
    def make_epoch_mag_dict(self):
        self.epoch_mag_dict = dict()
        self.get_trainable_layers()
        for curr_epoch in self.net_dict.keys():
            if is_number(curr_epoch):
                print(curr_epoch, end = "   ")
                self.get_net(curr_epoch)
                wd = self.make_weight_dict()
                mags = self.get_epoch_mags(wd)
                self.epoch_mag_dict[curr_epoch] = mags
        return None
    
    def save_epoch_mag_dict(self):
        if self.epoch_mag_dict is not None:
            pickle.dump(self.epoch_mag_dict, open(self.pkl_file,'wb'))
            print("Saved epoch_mag_dict to {}".format(self.pkl_file))
        
    def get_plot_info(self, trgt_layer):
        data_dict = dict()
        for curr_epoch in self.epoch_mag_dict.keys():
            curr_vecs = self.epoch_mag_dict[curr_epoch]
            if "conv" in trgt_layer:
                data_dict[curr_epoch] = [curr_vecs[trgt_layer][x]
                                         for x in curr_vecs[trgt_layer] ]
            elif "dense" in trgt_layer:
                data_dict[curr_epoch] = curr_vecs[trgt_layer]
            else:
                print("Unknown layer type")
                
        data = [data_dict[x] 
                for x in sorted(data_dict) 
                if x%10 == 0]
        labels = [x 
                  for x in sorted(data_dict) 
                  if x%10 == 0]
        return [data, labels]
    
    
    def plot_weights(self, data, labels, title, yscale='log'):

        fig, ax = plt.subplots(1,1,sharex=True,figsize=(15,4))

        ax.boxplot(data,
                   notch=False,  # notch shape
                   vert=True,  # vertical box alignment
                   patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks
        ax.set_xlabel('Source Training Epochs')
        ax.set_ylabel('Distribution Kernel Magnitudes')
        ax.set_yscale(yscale)
        ax.set_title("Layer: " + title)
        return fig
    
    def make_conv_pdf_graphs(self):
        plot_dict = OrderedDict()
        conv_layers = [x.name for x in self.trainable_layers if 'onv2d' in x.name]
        pp = PdfPages("conv_" + self.pdf_file)
        for trgt_layer in conv_layers:
            plot_dict[trgt_layer] = self.get_plot_info(trgt_layer)
            data, labels = plot_dict[trgt_layer]
            num_vecs = str(self.net_info[trgt_layer][1])
            vec_dim = str(np.prod(self.net_info[trgt_layer][0]))
            title = trgt_layer + " - " + num_vecs + "  " + vec_dim + "-dim Vectors"
            fig = self.plot_weights(data, labels, title)
            pp.savefig(fig)
        pp.close()
        
    def make_dense_pdf_graphs(self):
        plot_dict = OrderedDict()
        dense_layers = [x.name for x in self.trainable_layers if 'dense' in x.name]
        pp = PdfPages("dense_"+ self.pdf_file)
        for trgt_layer in dense_layers:
            plot_dict[trgt_layer] = self.get_plot_info(trgt_layer)
            data, labels = plot_dict[trgt_layer]
            vec_dim = str(np.prod(self.net_info[trgt_layer][0]))
            title = trgt_layer + "  " + vec_dim + "-dim Vector"
            fig = self.plot_dense_weights(data, labels, title)
            pp.savefig(fig)
        pp.close()

    def plot_dense_weights(self, data, labels, title):

        fig, ax = plt.subplots(1,1,sharex=True,figsize=(15,4))

        ax.plot(labels, data,color = 'steelblue',marker="o")
        ax.set_xlabel('Source Training Epochs')
        ax.set_ylabel('Magnitude of Weights Vector')
        ax.set_yscale('linear')
        ax.set_title("Layer: " + title)
        ax.set_xticks(np.arange(min(labels), max(labels)+1, 10.0))
        return fig

if __name__ == '__main__':

    home = os.path.expanduser('~')
    results_dir = os.path.join(home, 'Projects/opt-tfer-2/results', 'opt_tfer_expts',
                           'tinyimagenet200_notliving_living_expts', 'wide_resnet_28_10_arch',
                          'src_nets','workshop_expts','alt08.arl.army.mil_v0')

    print(os.listdir(results_dir))
    test = NetWeights(results_dir)
    test.make_epoch_mag_dict()
    test.save_epoch_mag_dict()
    test.make_conv_pdf_graphs()
    test.make_dense_pdf_graphs()

# python examine_layer_weights.py
# This code is still a little clunky. The results_dir variable
# needs to be hand adjusted to give the path to the directory holding
# snapshots of a net at various training epochs. Given that path
# this code will produce and save pdfs showing how the distribution
# of weights for each layer changes with training epoch. For conv layers
# it charts distributions of the magnitudes of weights forming each kernel
# (i.e. each kernel is considered to be a vector). For dense layers
# all the weights are combined to a single vector, whose weight is charted.
