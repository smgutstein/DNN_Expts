import argparse
from collections import defaultdict
from data_display import Data_Display
import importlib
import os
import pickle
import sys

file_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'dataset_loaders')
if file_dir not in sys.path:
    sys.path.append(file_dir)

dict_path = 'dataset_info/cifar100_dicts_all.pkl'
with open(dict_path,'r') as f:
    label_dict = pickle.load(f)
    
if __name__ == '__main__':

    # Get desired samples per class
    ap = argparse.ArgumentParser()
    ap.add_argument("dlm", type = str,
                    help="data loading module")
    args = ap.parse_args()

    # Import module to load desired data
    dlm = importlib.import_module(args.dlm)


    print("Loading data")
    (X_train, y_train), (X_test, y_test) = dlm.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.

    summary_dict = defaultdict(int)
    for curr in y_train:
        summary_dict[curr[0]]+=1

    for curr in sorted(summary_dict):
        print label_dict[curr],":",summary_dict[curr]

    print "\n---------------------------------------------\n"
    print "Total Samples:", y_train.shape[0]
    data_display = Data_Display(X_train, y_train, label_dict)
    data_display.start_display()
