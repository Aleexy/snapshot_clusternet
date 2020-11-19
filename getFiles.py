import csv
import os
import glob
import scipy.misc
import sys
import random
import numpy as np
import h5py
from sklearn.model_selection import train_test_split

mainpath = 'path to the data'
outpath = 'path for output'

if not os.path.exists(outpath):
    os.makedirs(outpath)

train_files = open(os.path.join(outpath, 'train_files.txt'), 'w')
test_files = open(os.path.join(outpath, 'test_files.txt'), 'w')

for root, dirs, files in os.walk(mainpath):
    for file in files:
        if file.endswith(".h5"):
            if 'train' in file:
                train_files.write(os.path.join(remote_path, file)+'\n')
            elif 'test' in file:
                test_files.write(os.path.join(remote_path, file)+'\n')

train_files.close()
test_files.close()
