import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import h5py
from os.path import join


class FabricsData(Dataset):
    """Data Handler for fabrics.

    Daniel note: unlike with other datasets, I'm subclassing torch.utils.data.Dataset.
    Why is that? Was `Dataset` added at a later date?

    Data is stored in `pure_random.hdf5` and when loaded, prints:

        <HDF5 dataset "actions": shape (7003, 15, 4), type "<f4">
        <HDF5 dataset "images": shape (7003, 16, 56, 56, 4), type "<f4">

    With 7003 episodes. However, we should have already split this into train and valid
    sets (80-20 split) beforehand. So this class just has to load and return values at
    certain indices. Slightly RAM-inefficient if we have to keep the data stored in RAM
    while training, but easier to implement.

    We should have run `python data/fabric_train_test.py` beforehand to generate pickle
    files, which are what we actually load.

    Update: we also support a second fabric data type.
    Update (Jan 20): using different data for each type for 95-5 split.
    """

    def __init__(self, train, data_root, seq_len, image_size=56, n_channels=4, use_actions=True):
        self.train = train
        self.data_root = data_root
        self.seq_len = seq_len
        self.image_size = image_size
        self.n_channels = n_channels
        self.use_actions = use_actions
        _, tail = os.path.split( data_root.rstrip('/') )

        # We already made pickle files with `fabric_train_test.py`. Adjust if needed.
        if self.train:
            if tail == 'fabric-random':
                path = join(data_root, 'pure_random_train_05602.pkl')  # 80-20
                #path = join(data_root, 'pure_random_train_06652.pkl')
            elif tail == 'fabric-01_2021':
                path = join(data_root, '01-2021_train_07945.pkl')  # 80-20
                #path = join(data_root, '01-2021_train_09435.pkl')
            else:
                raise ValueError(tail)
        else:
            if tail == 'fabric-random':
                path = join(data_root, 'pure_random_valid_01401.pkl')  # 80-20
                #path = join(data_root, 'pure_random_valid_00351.pkl')
            elif tail == 'fabric-01_2021':
                path = join(data_root, '01-2021_valid_01987.pkl')  # 80-20
                #path = join(data_root, '01-2021_valid_00497.pkl')
            else:
                raise ValueError(tail)

        with open(path, 'rb') as fh:
            data = pickle.load(fh)
            self.d_actions = data['actions']
            self.d_images = data['images']
            assert np.min(self.d_actions) >= -1.0, np.min(self.d_actions)
            assert np.max(self.d_actions) <=  1.0, np.max(self.d_actions)

        # Get number of items and episode length (for that use actions, NOT images)
        self.N = self.d_actions.shape[0]
        self.episode_len = self.d_actions.shape[1]

        # Double check.
        assert self.seq_len <= self.episode_len
        assert len(self.d_images.shape) == 5
        assert self.d_images.shape[0] == self.d_actions.shape[0]
        assert self.d_images.shape[1] == self.d_actions.shape[1] + 1
        assert self.d_images.shape[2] == self.d_images.shape[3] == image_size
        assert self.d_images.shape[4] == self.n_channels
        assert np.min(self.d_images) >= 0, np.min(self.d_images)
        assert np.max(self.d_images) <= 255.0, np.max(self.d_images)
        assert np.min(self.d_actions) >= -1.0, np.min(self.d_actions)
        assert np.max(self.d_actions) <=  1.0, np.max(self.d_actions)

        # Finally, divide by 255 so that values are in expected ranges.
        self.d_images = (self.d_images / 255.0).astype(np.float32)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        """Get image and (usually) action for training.

        NOTE: Extracting the way we do it means given an `index`, we do not know
        the exact properties of the data. Is it a concern? For now maybe not.

        Be very careful with indexing. Here, seq_len is context + output, and episode
        length is the number of actions (usually 15), so number of images is 15+1. We
        need to return all the possible images + actions that are needed for action
        conditioning to work. Examples with RSS 2020 params of context=3 and output=7:

          start=0 --> given o_{0,1,2}, a_{2,3,4,5,6,7,8},      predict o_{3:9}.
          start=3 --> given o_{3,4,5}, a_{5,6,7,8,9,10,11},    predict o_{6:12}.
          start=6 --> given o_{6,7,8}, a_{8,9,10,11,12,13,14}, predict o_{9:15}.

        We offset by 2, so 15-10+2 = 7 and we pick start index from 0 to 6. Returning
        the resulting sliced indices should give us all info we need. In SVG code, we
        will need to dump the first context-1 = 2 actions, since those are ignored in
        training, but all other actions (and all other images) are needed.

        ACTUALLY, upon further thought, since we want to concatenate the action with
        the output of the encoder, then it seems better to make `h` use a consistent
        dimension throughout training, so I think we want the first few actions, so do:

          start=0 --> given o_{0,1,2}, a_{0,1,2,3,4,5,6,7,8},      predict o_{3:9}.
          start=3 --> given o_{3,4,5}, a_{3,4,5,6,7,8,9,10,11},    predict o_{6:12}.
          start=6 --> given o_{6,7,8}, a_{6,7,8,9,10,11,12,13,14}, predict o_{9:15}.
        """
        seq_images = self.d_images[index]
        seq_actions = self.d_actions[index]

        # After extracting episode info, subsample a span of `seq_len = context+output`.
        start_idx = np.random.choice(self.episode_len - self.seq_len + 2)
        seq_images  = seq_images[ start_idx : start_idx + self.seq_len]
        seq_actions = seq_actions[start_idx : start_idx + self.seq_len - 1]

        if self.use_actions:
            return (seq_images, seq_actions)
        else:
            return seq_images
