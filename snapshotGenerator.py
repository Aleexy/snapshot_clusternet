# This file generates snapshots from a pointcloud dataset
import csv
import os
import glob
import scipy.misc
import sys
import random
import numpy as np
import statistics
import h5py
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


mainpath = 'path to the data'
outpath = 'path for output'

if not os.path.exists(outpath):
    os.makedirs(outpath)

NUM_POINT = 1024
COVERAGE = NUM_POINT*10
NUM_SAMPLES = 8000
NUM_SUBSAMPLING = 5
IS_TRAINING = False
whole_data = np.empty((0,3), dtype='float32')
whole_label = np.empty((0,1), int)

filenames = []

for root, dirs, files in os.walk(mainpath):
    for file in files:
        if file.endswith(".h5"):
             filenames.append(os.path.join(root, file))

for file in filenames:
    with h5py.File(file, 'r') as f:
        index = f['label'][()]
        data = np.float32(f['data'][()])
        whole_data = np.append(whole_data, data, axis = 0)
        whole_label = np.append(whole_label, index.reshape((-1, 1)), axis = 0)

def writeH5(data, label, count, is_train_data):
    coords = np.array(data).astype(np.float)
    shape = coords.shape
    if is_train_data:
        hdf5_path = outpath + 'train_' + str(label) + '_' + str(count) + '.h5'
    else:
        hdf5_path = outpath + 'test_' + str(label) + '_' + str(count) + '.h5'
    with h5py.File(hdf5_path, mode='w') as f:
        d = f.create_dataset('/data', data = coords)
        l = f.create_dataset('/label', data = label)

print('Fitting NearestNeighbors')
nbrs = NearestNeighbors(n_neighbors = COVERAGE, algorithm='kd_tree', leaf_size = 300).fit(whole_data)
print('Fitted')


############################## Generating Patches ##############################

#Dictionary for stats collections, needs to be modified based on the dataset
stats = {0: [0, 0],
         1: [0, 0],
         2: [0, 0],
         3: [0, 0],
         4: [0, 0],
         5: [0, 0],
         6: [0, 0],
         7: [0, 0]}

for count in range(0, NUM_SAMPLES, 5):
    rand_point = random.randint(0, len(whole_data)-1)
    rand_point = whole_data[rand_point].reshape(-1, 3)
    distances, indices = nbrs.kneighbors(rand_point, n_neighbors=COVERAGE)
    indices = list(indices[0])
    for subsample in range(NUM_SUBSAMPLING):
        sub_mask = random.sample(indices, NUM_POINT)
        new_obj = whole_data[sub_mask]
        (keys, counts) = np.unique(whole_label[sub_mask], return_counts=True)
        count_dict = dict(zip(keys, counts))
        major_class = keys[np.argmax(counts)]
        major_class_counts = count_dict[major_class]

        stats[major_class][0] += 1
        stats[major_class][1] += major_class_counts

        writeH5(new_obj, major_class, count+subsample, IS_TRAINING)
        print(count+subsample)

true_points = 0
for key in stats:
    if stats[key][0]:
        print(key, 'class has ', stats[key][1]/(stats[key][0] * NUM_POINT), '% true points')
        print(key, 'class has ', stats[key][0], 'samples')
        true_points += stats[key][1]

print('All samples has ', true_points/(NUM_SAMPLES * NUM_POINT), '% true points')
