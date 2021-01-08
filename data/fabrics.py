import socket
import numpy as np
from torchvision import datasets, transforms

class FabricsData(object):
    """Data Handler for fabrics.

    TODO
    """

    def __init__(self, train, data_root, seq_len=20, image_size=56):
        self.train = train
        self.data_root = data_root
        self.seq_len = seq_len
        self.image_size = image_size

        self.seed_is_set = False
        self.data = None  # TODO
        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        raise NotImplementedError()
