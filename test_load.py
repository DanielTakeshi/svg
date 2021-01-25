"""Put things in a class to make it easier to save/load in other projects."""
import argparse
import os
import sys
import pickle
import random
import numpy as np

# This is key. Actually should be lowercase, oops. :)
# We'll have to figure out how to import this in other code.
import SVG

# Here's how we can test loading.
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='')
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--n_future',  default=5)
opt = parser.parse_args()

# Might as well assign it here.
opt.model_dir = '/data/svg/logs/fabric-01_2021/model=dcgan56x56-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=2-n_future=5-lr=0.0020-g_dim=128-z_dim=10-last_frame_skip=False-beta=0.0001000-act-cond-1/model_0010.pth'

svg = SVG.SVG(opt)
print('\nSuccessfully loaded! Woo hoo, see the opt above. Now try prediction.')

# Load, usually from the cloth-visual-mpc/logs directory. For simplicity just hard-code something.
DATA_PATH = '/home/seita/cloth-visual-mpc/logs/demos-fabric-01-2021-epis_400_COMBINED.pkl'
with open(DATA_PATH, 'rb') as fh:
    pkl = pickle.load(fh)

# Take the first episode.
episode = pkl[0]
num_steps = len(episode['act'])
all_acts = np.array(episode['act'])
preds = []
acts = []

for i in range(num_steps + 1 - opt.n_future):
    currim = episode['obs'][i]                      # shape (56, 56, 4), type uint8
    curracts = all_acts[i:i + opt.n_future]         # shape (H, 4)
    pred = svg.predict(x=currim, x_acts=curracts)   # shape (H, 56, 56, 4), float32, in (0,1)
    preds.append(pred)                              # ALL preds: \hat{x}_{i+1 : i+H}
    acts.append(curracts)                           # input actions: a_{i : i+H-1}

# With my way of saving, shapes are (N, H, 56, 56, 4) and (N, H, 4)
# where N=num_steps-H+1. (All predictions given this episode + horizon)
gt_obs_all = np.array(episode['obs']).astype(np.uint8)
preds_all = np.array(preds) * 255.0     # NOTE: multiply by 255 since output is all in (0,1).
preds_all = preds_all.astype(np.uint8)
print('done with predictions!')