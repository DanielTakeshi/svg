"""
I think it's easier to split the hdf5 file beforehand, so we get train and
test splits that we use. Then loading fabric data is a breeze.
Note: requires protocol=4 for large pickle file dumping.
Example output:

~/svg $ python data/fabric_train_test.py
N: 7003, len idxs_t,idxs_v: 5602, 1401
train image shape (5602, 16, 56, 56, 4) action: (5602, 15, 4)
valid image shape (1401, 16, 56, 56, 4) action: (1401, 15, 4)
	type img: float32 action: float32
	type img: float32 action: float32

Some data statistics:
 tr  min/max/mean/medi: 0.00, 255.00, 135.992, 126.00
 val min/max/mean/medi: 0.00, 255.00, 136.662, 127.00

Now for RGBD channels (c=3 means depth):
 c0  min/max/mean: 0.00, 255.00, 149.41
 c0  min/max/mean: 0.00, 255.00, 150.17
 c1  min/max/mean: 0.00, 255.00, 128.35
 c1  min/max/mean: 0.00, 255.00, 128.60
 c2  min/max/mean: 0.00, 255.00, 111.73
 c2  min/max/mean: 0.00, 255.00, 112.82
 c3  min/max/mean: 0.00, 255.00, 154.48
 c3  min/max/mean: 0.00, 255.00, 155.06

Now actions:
 tr  acts min/max/mean: -1.00, 1.00, -0.0037
 val acts min/max/mean: -1.00, 1.00, 0.0002
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
print('train image shape', X_t.shape, 'action:', d_train['actions'].shape)
print('valid image shape', X_v.shape, 'action:', d_valid['actions'].shape)
print('\ttype img:', X_t.dtype, 'action:', d_train['actions'].dtype)
print('\ttype img:', X_v.dtype, 'action:', d_valid['actions'].dtype)
print('\nSome data statistics:')
print(' tr  min/max/mean/medi: {:0.2f}, {:0.2f}, {:0.3f}, {:0.2f}'.format(
        X_t.min(), X_t.max(), X_t.mean(), np.median(X_t)))
print(' val min/max/mean/medi: {:0.2f}, {:0.2f}, {:0.3f}, {:0.2f}'.format(
        X_v.min(), X_v.max(), X_v.mean(), np.median(X_v)))
print('\nNow for RGBD channels (c=3 means depth):')
for c in range(4):
    print(' c{}  min/max/mean: {:0.2f}, {:0.2f}, {:0.2f}'.format(c,
            np.min( X_t[:,:,:,:,c]),
            np.max( X_t[:,:,:,:,c]),
            np.mean(X_t[:,:,:,:,c])))
    print(' c{}  min/max/mean: {:0.2f}, {:0.2f}, {:0.2f}'.format(c,
            np.min( X_v[:,:,:,:,c]),
            np.max( X_v[:,:,:,:,c]),
            np.mean(X_v[:,:,:,:,c])))
print('\nNow actions:')
print(' tr  acts min/max/mean: {:0.2f}, {:0.2f}, {:0.4f}'.format(
        d_train['actions'].min(), d_train['actions'].max(), d_train['actions'].mean()))
print(' val acts min/max/mean: {:0.2f}, {:0.2f}, {:0.4f}'.format(
        d_valid['actions'].min(), d_valid['actions'].max(), d_valid['actions'].mean()))

with open(pth_valid, 'wb') as fh:
    pickle.dump(d_valid, fh, protocol=4)
with open(pth_train, 'wb') as fh:
    pickle.dump(d_train, fh, protocol=4)
