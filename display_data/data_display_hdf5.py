from collections import defaultdict
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import numpy as np

class Data_Display_hdf5(object):

    def __init__(self, image_generator,
                 label_dict):
        self.image_generator = image_generator
        self.label_dict = label_dict
        self.curr_batch = self.get_next_batch()

        self.images = self.curr_batch[0]
        self.labels = self.curr_batch[1]
        self.batch_size = self.images.shape[0]

        self.ctr = 0
        self.currClass = 0

        self.fig, self.ax = plt.subplots(1)
        self.ax.axes.get_xaxis().set_visible(False)
        self.ax.axes.get_yaxis().set_visible(False)
        self.axprev = plt.axes([0.1, 0.025, 0.1, 0.05])
        self.axnext = plt.axes([0.85, 0.025, 0.1, 0.05])
        
        self.bnext = Button(self.axnext, 'Next')
        self.bnext.on_clicked(self.next)
        
        self.bprev = Button(self.axprev, 'Previous')
        self.bprev.on_clicked(self.prev)

    def get_next_batch(self):

        data, labels = self.image_generator.next()
        data = data.astype('float32')

        import pdb
        pdb.set_trace()
        if data.max()> 1.5:
            data /= 255.
        if data.max()> 1.5:        
            data /= 255.

        summary_dict = defaultdict(int)
        for curr in labels:
            summary_dict[curr]+=1

        for curr in sorted(summary_dict):
            print (self.label_dict[curr],":",summary_dict[curr])


        return (data, labels)

        
    def start_display(self):
        self.show_image()
        #plt.ion()
        plt.show()

    def show_image(self):
        self.ax.clear()
        self.curr_im = self.images[self.ctr,:,:,:]

        self.category_num = self.labels[self.ctr]
        self.category_str = self.label_dict[self.category_num]
        self.category_str += ' (' + str(self.category_num) + ')'

        self.ax.set_title(self.category_str)
        self.ax.imshow(self.curr_im)
                

    def next(self, event):
        self.ctr += 1
        if self.ctr >= self.batch_size:
           self.ctr = 0
        self.show_image()
            
    def prev(self, event):
        self.ctr -= 1
        if self.ctr < 0:
           self.ctr = self.batch_size - 1
        self.show_image()



