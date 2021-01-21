"""
Compare SV2P and SVG. [SV2P is trained elsewhere, but the predictions file we use
will be in a standard pickle file that we can load anywhere.]

For SV2P prediction format, see:
https://github.com/ryanhoque/cloth-visual-mpc/blob/sv2p/vismpc/scripts/predict.py
"""
import os
import cv2
import sys
import pickle
import random
import argparse
import utils
import numpy as np
from os.path import join
from collections import defaultdict
from skimage.measure import compare_psnr as psnr_metric
from skimage.measure import compare_ssim as ssim_metric

# ------------------------------- DATA FILES ---------------------------------- #
HEAD = '/home/seita/cloth-visual-mpc/logs/'
TAIL_SV2P_01 = 'demos-2021-01-20-pol-random-seeds-987654_to_987657-actclip-domrand-rgbd-tiers_all-epis_400_COMBINED_PREDS_SV2P_01_masks.pkl'
TAIL_SV2P_10 = 'demos-2021-01-20-pol-random-seeds-987654_to_987657-actclip-domrand-rgbd-tiers_all-epis_400_COMBINED_PREDS_SV2P_10_masks.pkl'
SV2P_01_fabric_random_pth = join(HEAD, TAIL_SV2P_01)
SV2P_10_fabric_random_pth = join(HEAD, TAIL_SV2P_10)
with open(SV2P_01_fabric_random_pth, 'rb') as fh:
    SV2P_01_fabric_random = pickle.load(fh)
with open(SV2P_10_fabric_random_pth, 'rb') as fh:
    SV2P_10_fabric_random = pickle.load(fh)

# TODO(daniel)
HEAD = '/home/seita/svg/logs/'
PRED_SVG = 'TODO'
# ------------------------------- DATA FILES ---------------------------------- #


def compute_ssim_metrics():
    pass


def save_images():
    pass


if __name__ == "__main__":
    #compute_ssim_metrics()
    save_images()
