"""
I think it's easier to split the hdf5 file beforehand, so we get train and
test splits that we use. Then loading fabric data is a breeze.
Note: requires protocol=4 for large pickle file dumping.

Run: python data/fabric_train_test.py

---------------------------------------------------------------------------------
Output from the RSS 2020 data:
(UPDATE Jan 20: this is with the 80-20 data, for 95-5 I converted to uint8.)
---------------------------------------------------------------------------------

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

---------------------------------------------------------------------------------
Output from the 01-2021 data:
FYI this has episode lengths of 10. Images are uint8 to save space.
The action magnitudes are also larger at the extremes.
UPDATE: after discussion with Ryan, I clip the actions here.
---------------------------------------------------------------------------------

N: 9932, len idxs_t,idxs_v: 7945, 1987
train image shape (7945, 11, 56, 56, 4) action: (7945, 10, 4)
valid image shape (1987, 11, 56, 56, 4) action: (1987, 10, 4)
	type img: uint8 action: float32
	type img: uint8 action: float32

Some data statistics:
 tr  min/max/mean/medi: 0.00, 255.00, 137.347, 144.00
 val min/max/mean/medi: 0.00, 255.00, 137.132, 144.00

Now for RGBD channels (c=3 means depth):
 c0  min/max/mean: 0.00, 255.00, 128.85
 c0  min/max/mean: 0.00, 255.00, 128.79
 c1  min/max/mean: 0.00, 255.00, 130.96
 c1  min/max/mean: 0.00, 255.00, 131.14
 c2  min/max/mean: 0.00, 255.00, 138.74
 c2  min/max/mean: 0.00, 255.00, 138.09
 c3  min/max/mean: 0.00, 228.00, 150.84
 c3  min/max/mean: 0.00, 228.00, 150.51

Now actions:
 tr  acts min/max/mean: -1.37, 1.37, -0.0004
 val acts min/max/mean: -1.35, 1.35, -0.0005

---------------------------------------------------------------------------------
We may do some other datasets that combine the above. I'm also trying one with a
95-5 split between train-valid to see if that helps. BTW for combining them we
DO NOT want to concatenate numpy arrays as they have different episode lengths.
Better to keep them separate, and in the PyTorch data loader we can handle this
logic of handling two arrays.
---------------------------------------------------------------------------------
"""
import os
import numpy as np
import h5py
import pickle

# Pick the data type and just run `python data/fabric_train_test.py`.
TYPE = 'fabric-random'
#TYPE = 'fabric-01_2021'
FRAC_TRAIN = 0.95

# Note that the fabric-random data was saved with float32s.
if TYPE == 'fabric-random':
    ROOT      = '/data/svg/fabric-random/'
    DATA_ROOT = os.path.join(ROOT, 'pure_random.hdf5')
elif TYPE == 'fabric-01_2021':
    ROOT      = '/data/svg/fabric-01_2021/'
    DATA_ROOT = os.path.join(ROOT, 'data.hdf5')
else:
    raise ValueError(TYPE)


def debug(d_train, d_valid):
    """Debugging."""
    X_t = d_train['images']
    X_v = d_valid['images']
    print(f'Num episodes: {N}, len idxs_t,idxs_v: {len(idxs_t)}, {len(idxs_v)}')
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


if __name__ == "__main__":
    # Copy hdf5 files to numpy arrays with `[:]`. Convert to np.uint8.
    with h5py.File(DATA_ROOT, 'r') as f:
        d_actions = (f['actions'])[:]
        d_images  = (f['images'])[:]
    d_images = d_images.astype(np.uint8)

    # Get indices for train and validation.
    N = len(d_images)
    num_t = int(FRAC_TRAIN * N)
    num_v = N - num_t
    indices = np.random.permutation(N)
    idxs_t = indices[:num_t]
    idxs_v = indices[num_t:]

    # File names.
    if TYPE == 'fabric-random':
        pth_train = os.path.join(ROOT, f'pure_random_train_{str(num_t).zfill(5)}.pkl')
        pth_valid = os.path.join(ROOT, f'pure_random_valid_{str(num_v).zfill(5)}.pkl')
    elif TYPE == 'fabric-01_2021':
        pth_train = os.path.join(ROOT, f'01-2021_train_{str(num_t).zfill(5)}.pkl')
        pth_valid = os.path.join(ROOT, f'01-2021_valid_{str(num_v).zfill(5)}.pkl')
    else:
        raise ValueError(TYPE)

    # Split up the data, debug, and clip actions.
    d_train = {'images': d_images[idxs_t], 'actions': d_actions[idxs_t]}
    d_valid = {'images': d_images[idxs_v], 'actions': d_actions[idxs_v]}
    debug(d_train, d_valid)
    d_train['actions'] = np.clip(d_train['actions'], a_min=-1.0, a_max=1.0)
    d_valid['actions'] = np.clip(d_valid['actions'], a_min=-1.0, a_max=1.0)

    # Save as pickle files.
    with open(pth_valid, 'wb') as fh:
        pickle.dump(d_valid, fh, protocol=4)
    with open(pth_train, 'wb') as fh:
        pickle.dump(d_train, fh, protocol=4)
    print('\nLook at this for the saved data:')
    print(f'{pth_valid}')
    print(f'{pth_train}')
