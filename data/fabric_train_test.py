"""
I think it's easier to split the hdf5 file beforehand, so we get train and
test splits that we use. Then loading fabric data is a breeze. Example output:

~/svg $ python data/fabric_train_test.py
N: 7003, len idxs_t,idxs_v: 5602, 1401
train (5602, 16, 56, 56, 4) (5602, 15, 4)
valid (1401, 16, 56, 56, 4) (1401, 15, 4)
	 float32 float32
	 float32 float32

Note: requires protocol=4 for large pickle file dumping.
"""
import socket
import numpy as np
import h5py
import pickle

data_root = '/data/svg/fabric-random/pure_random.hdf5'

# Copy hdf5 files to numpy arrays with `[:]`.
with h5py.File(data_root, 'r') as f:
    d_actions = (f['actions'])[:]
    d_images = (f['images'])[:]

# Get indices for train and validation.
N = len(d_images)
num_t = int(0.8 * N)
num_v = N - num_t
indices = np.random.permutation(N)
idxs_t = indices[:num_t]
idxs_v = indices[num_t:]

# File names.
pth_train = f'/data/svg/fabric-random/pure_random_train_{str(num_t).zfill(5)}.pkl'
pth_valid = f'/data/svg/fabric-random/pure_random_valid_{str(num_v).zfill(5)}.pkl'

# Split up the data.
data_train = {'images': d_images[idxs_t], 'actions': d_actions[idxs_t]}
data_valid = {'images': d_images[idxs_v], 'actions': d_actions[idxs_v]}

# Debugging
print(f'N: {N}, len idxs_t,idxs_v: {len(idxs_t)}, {len(idxs_v)}')
print('train', data_train['images'].shape, data_train['actions'].shape)
print('valid', data_valid['images'].shape, data_valid['actions'].shape)
print('\t', data_train['images'].dtype, data_train['actions'].dtype)
print('\t', data_valid['images'].dtype, data_valid['actions'].dtype)

with open(pth_valid, 'wb') as fh:
    pickle.dump(data_valid, fh, protocol=4)
with open(pth_train, 'wb') as fh:
    pickle.dump(data_train, fh, protocol=4)
