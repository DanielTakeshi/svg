"""
I think it's easier to split the hdf5 file beforehand, so we get train and
test splits that we use. Then loading fabric data is a breeze. Example output:

~/svg $ python data/fabric_train_test.py
N: 7003, len idxs_t,idxs_v: 5602, 1401
train (5602, 16, 56, 56, 4) (5602, 16, 56, 56, 4)
valid (1401, 16, 56, 56, 4) (1401, 16, 56, 56, 4)
	type: float32 float32
	type: float32 float32

Some data statistics:
 tr  min/max/mean: 0.00, 255.00, 136.17
 val min/max/mean: 0.00, 255.00, 135.94
Now for RGBD channels:
 c0  min/max/mean: 0.00, 255.00, 149.62
 c0  min/max/mean: 0.00, 255.00, 149.33
 c1  min/max/mean: 0.00, 255.00, 128.41
 c1  min/max/mean: 0.00, 255.00, 128.35
 c2  min/max/mean: 0.00, 255.00, 112.06
 c2  min/max/mean: 0.00, 255.00, 111.52
 c3  min/max/mean: 0.00, 255.00, 154.60
 c3  min/max/mean: 0.00, 255.00, 154.57

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
d_train = {'images': d_images[idxs_t], 'actions': d_actions[idxs_t]}
d_valid = {'images': d_images[idxs_v], 'actions': d_actions[idxs_v]}

# Debugging
X_t = d_train['images']
X_v = d_valid['images']
print(f'N: {N}, len idxs_t,idxs_v: {len(idxs_t)}, {len(idxs_v)}')
print('train',   X_t.shape, X_t.shape)
print('valid',   X_v.shape, X_v.shape)
print('\ttype:', X_t.dtype, X_t.dtype)
print('\ttype:', X_v.dtype, X_v.dtype)
print('\nSome data statistics:')
print(' tr  min/max/mean: {:0.2f}, {:0.2f}, {:0.2f}'.format(X_t.min(), X_t.max(), X_t.mean()))
print(' val min/max/mean: {:0.2f}, {:0.2f}, {:0.2f}'.format(X_v.min(), X_v.max(), X_v.mean()))
print('Now for RGBD channels:')
for c in range(4):
    print(' c{}  min/max/mean: {:0.2f}, {:0.2f}, {:0.2f}'.format(c,
            np.min( X_t[:,:,:,:,c]),
            np.max( X_t[:,:,:,:,c]),
            np.mean(X_t[:,:,:,:,c])))
    print(' c{}  min/max/mean: {:0.2f}, {:0.2f}, {:0.2f}'.format(c,
            np.min( X_v[:,:,:,:,c]),
            np.max( X_v[:,:,:,:,c]),
            np.mean(X_v[:,:,:,:,c])))

with open(pth_valid, 'wb') as fh:
    pickle.dump(d_valid, fh, protocol=4)
with open(pth_train, 'wb') as fh:
    pickle.dump(d_train, fh, protocol=4)
