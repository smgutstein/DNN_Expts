import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button


class Data_Display(object):

    def __init__(self, images, labels,
                 label_dict):
        self.images = images
        self.labels = labels
        self.label_dict = label_dict

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

        
    def start_display(self):
        self.show_image()
        #plt.ion()
        plt.show()

    def show_image(self):
        self.ax.clear()
        self.curr_im = self.images[self.ctr,:,:,:]

        self.category_num = self.labels[self.ctr][0]
        self.category_str = self.label_dict[self.category_num]
        self.category_str += ' (' + str(self.category_num) + ')'

        self.ax.set_title(self.category_str)
        self.ax.imshow(self.curr_im)
                

    def next(self, event):
        self.ctr += 1
        if self.ctr >= self.labels.shape[0]:
           self.ctr = 0
        self.show_image()
            
    def prev(self, event):
        self.ctr -= 1
        if self.ctr < 0:
           self.ctr = self.labels.shape[0] - 1
        self.show_image()



