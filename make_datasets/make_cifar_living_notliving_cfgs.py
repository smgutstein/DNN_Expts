import configparser
import itertools
import os

config = configparser.ConfigParser()

path = "./cfg_dir2/expt_cfg/tfer_expts/opt_tfer_series"
prefix = 'cifar100_nonliving_'
midfix = '_trgt_iter_'
suffix = '.cfg'

SPC = '10spc'
TR_ID = 'a'
SRC_ITER = '0'
base_file = os.path.join(path, SPC, prefix + TR_ID + midfix + SRC_ITER + suffix)
config.read(base_file)


SPC_list = [ str(x) for x in [10,25,50,75,100,125,150,175,200,250]]
TR_ID_list = ['a', 'b', 'c', 'd', 'e']
SRC_ITER_list = [str(x) for x in [0,10,20,25,30,40,50,60,70,75,80,90,100,125,150,175,200,250,'best']]
tot_list = [SPC_list, TR_ID_list, SRC_ITER_list]

outroot = './cfg_dir2/expt_cfg/tfer_expts/opt_tfer_series_2'
for curr_SPC, curr_TR_ID, curr_SRC_ITER in list(itertools.product(*tot_list)):
    outfile_name = prefix + curr_SPC + midfix + curr_SRC_ITER + '.cfg'
    print("Curr Outfile: ", outfile_name)
    
    temp = config['ExptFiles']['expt_subdir'].split('/')[:-1]
    if 'batch' not in temp[-1]:
        temp[-1] += '_batch'
    temp = '/'.join(temp + [curr_SPC + 'spc'])
    config['ExptFiles']['expt_subdir'] = temp
    
    temp1 = config['TrgtTaskParams']['data_loader'].split('_')[:-2]
    temp1 = "_".join(temp1 + [curr_SPC, curr_TR_ID])
    config['TrgtTaskParams']['data_loader'] = temp1
    
    config['SavedParams']['saved_iter'] = curr_SRC_ITER
    
    outname = prefix + curr_TR_ID + midfix + curr_SRC_ITER + suffix
    outpath = os.path.join(outroot, curr_SPC+'spc')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    with open(os.path.join(outpath, outname), 'w') as f:
        config.write(f)
