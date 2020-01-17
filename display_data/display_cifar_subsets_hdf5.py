import argparse
from data_display_hdf5 import Data_Display_hdf5
import importlib
import os
import pickle
import sys

file_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'dataset_loaders')
if file_dir not in sys.path:
    sys.path.append(file_dir)

dict_path = 'dataset_info/cifar100_dicts_all.pkl'
with open(dict_path,'rb') as f:
    label_dict = pickle.load(f, encoding='latin1')
    
if __name__ == '__main__':

    # Get desired samples per class
    ap = argparse.ArgumentParser()
    ap.add_argument("dgm", type = str,
                    help="data generating module")
    args = ap.parse_args()

    # Import module to load desired data
    dgm = importlib.import_module(args.dgm)


    print("Loading data")
    (trainGen, testGen, info, (depth, height, width)) = dgm.get_generator(test=True)
    data_display = Data_Display_hdf5(trainGen, label_dict)
    data_display.start_display()
    

# e.g. python display_cifar_subsets_hdf5.py cifar_augment_generator5
