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
import PIL
from PIL import (Image, ImageDraw)
HORIZON = 5
RESULT_DIR = 'results'

# ----------------------------------------------------------------------------- #
# ------------------------------- DATA FILES ---------------------------------- #
# ----------------------------------------------------------------------------- #
HEAD = '/home/seita/cloth-visual-mpc/logs/'
TAIL_SV2P_01 = 'demos-2021-01-20-pol-random-seeds-987654_to_987657-actclip-domrand-rgbd-tiers_all-epis_400_COMBINED_PREDS_SV2P_01_masks.pkl'
TAIL_SV2P_10 = 'demos-2021-01-20-pol-random-seeds-987654_to_987657-actclip-domrand-rgbd-tiers_all-epis_400_COMBINED_PREDS_SV2P_10_masks.pkl'

# fr = fabric_random
SV2P_01_fr_pth = join(HEAD, TAIL_SV2P_01)
SV2P_10_fr_pth = join(HEAD, TAIL_SV2P_10)

# These are _lists_, one dict per episode.
with open(SV2P_01_fr_pth, 'rb') as fh:
    SV2P_01_fr = pickle.load(fh)
with open(SV2P_10_fr_pth, 'rb') as fh:
    SV2P_10_fr = pickle.load(fh)
assert len(SV2P_01_fr) == len(SV2P_10_fr)

# TODO(daniel)
HEAD = '/home/seita/svg/logs/'
PRED_SVG = 'TODO'
# ----------------------------------------------------------------------------- #
# ------------------------------- DATA FILES ---------------------------------- #
# ----------------------------------------------------------------------------- #


def depth_to_3ch(img, cutoff):
    """Useful to turn the background into black into the depth images.
    """
    w,h = img.shape
    new_img = np.zeros([w,h,3])
    img = img.flatten()
    img[img>cutoff] = 0.0
    img = img.reshape([w,h])
    for i in range(3):
        new_img[:,:,i] = img
    return new_img


def depth_scaled_to_255(img):
    assert np.max(img) > 0.0
    img = 255.0/np.max(img)*img
    img = np.array(img,dtype=np.uint8)
    for i in range(3):
        img[:,:,i] = cv2.equalizeHist(img[:,:,i])
    return img


def make_depth_img(img, cutoff=1000):
    # NOTE! Depth will look noisy due to domain randomization.
    img = np.squeeze(img)
    img = depth_to_3ch(img, cutoff)
    img = depth_scaled_to_255(img)
    return img


def compute_ssim_metrics():
    pass


def save_images():
    """Collect a bunch of images of ground truth sequences followed by predictions.

    Ideally we can then paste these together for a figure.
    BTW the ground truths should be the same among the different models, since we
    tested on the same data, so we can do another sanity check. Same for actions.
    All of these should be with context of 1, and predicting the next 5 frames.

    BTW: to make a figure we can just take screenshots of appropriate segments?
    """

    for ep in range(len(SV2P_01_fr)):
        if ep % 25 == 0:
            print(f'checking predictions on episode {ep}')
        # Predictions have shape (N, H=HORIZON, 56, 56, 4), N is number of predictions.
        gt_obs          = SV2P_01_fr[ep]['gt_obs']
        gt_obs_0        = SV2P_10_fr[ep]['gt_obs']
        gt_context      = SV2P_01_fr[ep]['contexts']
        gt_context_0    = SV2P_10_fr[ep]['contexts']
        gt_act          = SV2P_01_fr[ep]['act']
        gt_act_0        = SV2P_10_fr[ep]['act']
        pred_sv2p_01_fr = SV2P_01_fr[ep]['pred']
        pred_sv2p_10_fr = SV2P_10_fr[ep]['pred']

        # We load multiple versions partly as an extra sanity check.
        assert gt_obs.shape == gt_obs_0.shape
        assert gt_context.shape == gt_context_0.shape
        assert gt_act.shape == gt_act_0.shape
        assert np.allclose(gt_act, gt_act_0)

        # Actually cannot use len(gt_act), that's not the number of actions in an episode
        # Oh, the N can be negative so if that's the case shape of gt_context is (0,)
        num_obs = len(gt_obs)
        N = num_obs - HORIZON
        assert N <= gt_context.shape[0], f'{N} vs {gt_context.shape}'

        # New image for each episode? Actually maybe consider splitting depth vs color?
        # num_steps is number of actions but also need to add padding
        ws = 5                          # whitespace padding
        IM_WIDTH  = (num_obs) * 56 + (num_obs+1) * ws

        # What do we want in the image? 2 rows for GT color and depth, then 2 for each
        # prediction of SV2P (1 and 10 masks) on the RGB image? N = number of predictions.
        # Then do 1 more row after that just to give empty breathing room to make sure we
        # did this right ... (sanity checks). Update: adding 2 more for depth.
        nrows = 2 + (N * 4)
        nrows = max(3, nrows)  # handle negative N
        IM_HEIGHT = (nrows+1)*ws + (nrows*56)

        # Finally make this PIL Image.
        t_img = Image.new(mode='RGB', size=(IM_WIDTH,IM_HEIGHT), color=(255,255,255))
        draw = ImageDraw.Draw(t_img)

        # Ground truth sequence at the top
        for t in range(num_obs):
            _w = (t * 56) + ((t+1) * ws)
            _h = ws
            frame_rgb = gt_obs[t, :, :, :3]
            frame_d   = gt_obs[t, :, :, 3:]
            frame_d   = make_depth_img(frame_d)
            t_img.paste(PIL.Image.fromarray(frame_rgb), (_w, _h)          )
            t_img.paste(PIL.Image.fromarray(frame_d),   (_w, _h + 56 + ws))

        # Iterate through all predictions, adding them sequentially. Update: actually
        # we have to check to see if the episode has enough length at all. If an episode
        # has under 5 actions then we could not do prediction on it.
        row = 2
        hoff = 0

        for t in range(0, N):
            # NOTE: for screenshotting here for figures, we can reorder these, so long
            # as we check that the rows are being incremented appropriately.
            sv2p_01 = pred_sv2p_01_fr[t,:,:,:,:]  # (H,56,56,4)
            sv2p_10 = pred_sv2p_10_fr[t,:,:,:,:]  # (H,56,56,4)
            context   = gt_context[t,:,:,:]
            context_0 = gt_context_0[t,:,:,:]

            # SV2P ONE MASK. Context, then predicted images.
            img = PIL.Image.fromarray(context[:,:,:3])
            _w = ws + hoff
            _h = (row+1)*ws + row*56
            t_img.paste(img, (_w, _h))
            for h in range(HORIZON):
                img = PIL.Image.fromarray(sv2p_01[h, :, :, :3])
                _w = (h+2)*ws + (h+1)*56 + hoff
                _h = (row+1)*ws + row*56
                t_img.paste(img, (_w, _h))

            # Next row, SV2P 10 MASK
            row += 1
            img = PIL.Image.fromarray(context_0[:,:,:3])
            _w = ws + hoff
            _h = (row+1)*ws + row*56
            t_img.paste(img, (_w, _h))
            for h in range(HORIZON):
                img = PIL.Image.fromarray(sv2p_10[h, :, :, :3])
                _w = (h+2)*ws + (h+1)*56 + hoff
                _h = (row+1)*ws + row*56
                t_img.paste(img, (_w, _h))

            # SV2P ONE MASK, DEPTH.
            row += 1
            img = PIL.Image.fromarray(
                    make_depth_img(context[:,:,3:]))
            _w = ws + hoff
            _h = (row+1)*ws + row*56
            t_img.paste(img, (_w, _h))
            for h in range(HORIZON):
                img = PIL.Image.fromarray(
                        make_depth_img(sv2p_01[h, :, :, 3:]))
                _w = (h+2)*ws + (h+1)*56 + hoff
                _h = (row+1)*ws + row*56
                t_img.paste(img, (_w, _h))

            # Next row, SV2P 10 MASK, DEPTH
            row += 1
            img = PIL.Image.fromarray(
                    make_depth_img(context_0[:,:,3:]))
            _w = ws + hoff
            _h = (row+1)*ws + row*56
            t_img.paste(img, (_w, _h))
            for h in range(HORIZON):
                img = PIL.Image.fromarray(
                        make_depth_img(sv2p_10[h, :, :, 3:]))
                _w = (h+2)*ws + (h+1)*56 + hoff
                _h = (row+1)*ws + row*56
                t_img.paste(img, (_w, _h))

            # Prep for next row in next for loop.
            row += 1
            hoff += (ws + 56)

        # Save in BGR format, which means saving with OpenCV.
        estr = str(ep).zfill(3)
        sstr = str(num_obs).zfill(2)
        img_path = f'preds_ep_{estr}_obs_{sstr}_rgbd.png'
        img_path = join(RESULT_DIR, img_path)
        t_img_np = np.array(t_img)
        cv2.imwrite(img_path, t_img_np)


if __name__ == "__main__":
    #compute_ssim_metrics()
    save_images()