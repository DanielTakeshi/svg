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
from PIL import (Image, ImageDraw, ImageFont)
HORIZON = 5

# ----------------------------------------------------------------------------- #
# ------------------------------- DATA FILES ---------------------------------- #
# ----------------------------------------------------------------------------- #
HEAD_SV2P = '/home/seita/cloth-visual-mpc/logs/'
HEAD_SVG  = '/home/seita/svg/results_svg/'

# fr = fabric_random
TAIL_SV2P_01_fr = 'demos-fabric-random-epis_400_COMBINED_PREDS_SV2P_01.pkl'
TAIL_SV2P_10_fr = 'demos-fabric-random-epis_400_COMBINED_PREDS_SV2P_10.pkl'
TAIL_SVG_fr     = 'demos-fabric-random-epis_400_COMBINED_PREDS_SVG-LP.pkl'
SV2P_01_fr_pth  = join(HEAD_SV2P, TAIL_SV2P_01_fr)
SV2P_10_fr_pth  = join(HEAD_SV2P, TAIL_SV2P_10_fr)
SVG_fr_pth      = join(HEAD_SVG,  TAIL_SVG_fr)

# fnew = fabric_01-2021 (new data)
TAIL_SV2P_10_fnew = 'demos-fabric-01-2021-epis_400_COMBINED_PREDS_SV2P_10.pkl'
TAIL_SVG_fnew     = 'demos-fabric-01-2021-epis_400_COMBINED_PREDS_SVG-LP.pkl'
SV2P_10_fnew_pth  = join(HEAD_SV2P, TAIL_SV2P_10_fnew)
SVG_fnew_pth      = join(HEAD_SVG,  TAIL_SVG_fnew)

# These are _lists_, one dict per episode.
with open(SV2P_01_fr_pth, 'rb') as fh:
    SV2P_01_fr = pickle.load(fh)
with open(SV2P_10_fr_pth, 'rb') as fh:
    SV2P_10_fr = pickle.load(fh)
with open(SVG_fr_pth, 'rb') as fh:
    SVG_fr = pickle.load(fh)

# fnew data
with open(SV2P_10_fnew_pth, 'rb') as fh:
    SV2P_10_fnew = pickle.load(fh)
with open(SVG_fnew_pth, 'rb') as fh:
    SVG_fnew = pickle.load(fh)

assert len(SV2P_01_fr) == len(SV2P_10_fr) == len(SVG_fr) == len(SV2P_10_fnew) == len(SVG_fnew)
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


def draw_ssim(draw, IM_WIDTH, ws, hoff, groupname):
    # Draw some structural similarity values as a sanity check.
    fontsize = 10
    ssim_width = IM_WIDTH - (ws + 56)  # Make this LAST column
    text_0 = 'ssim_0'
    text_1 = 'ssim_1'
    text_2 = 'ssim_2'
    text_3 = 'ssim_3'
    text_4 = 'ssim_4'
    off = 9
    draw.text((ssim_width, hoff+0*off), groupname, fill=(0,0,0,0))
    draw.text((ssim_width, hoff+1*off), text_0, fill=(0,0,0,0))
    draw.text((ssim_width, hoff+2*off), text_1, fill=(0,0,0,0))
    draw.text((ssim_width, hoff+3*off), text_2, fill=(0,0,0,0))
    draw.text((ssim_width, hoff+4*off), text_3, fill=(0,0,0,0))
    draw.text((ssim_width, hoff+5*off), text_4, fill=(0,0,0,0))


def save_images_get_ssim(datatype):
    """Collect a bunch of images of ground truth sequences followed by predictions.
    Also, might as well compute SSIM since everything is already nicely loaded.

    Ideally we can then paste these together for a figure.
    BTW the ground truths should be the same among the different models, since we
    tested on the same data, so we can do another sanity check. Same for actions.
    All of these should be with context of 1, and predicting the next 5 frames.

    BTW: to make a figure we can just take screenshots of appropriate segments?
    Might make it easier since I'm grouping everything anyway.
    """
    assert datatype in ['fabric-random', 'fabric-01-2021'], datatype

    for ep in range(len(SV2P_01_fr)):
        if ep % 25 == 0:
            print(f'checking predictions on episode {ep}')

        # Predictions have shape (N, H=HORIZON, 56, 56, 4), N is number of predictions.
        # I load a bunch of un-necessary stuff, just make sure 'pred' is correct!
        if datatype == 'fabric-random':
            gt_obs          = SV2P_01_fr[ep]['gt_obs']
            gt_obs_0        = SV2P_10_fr[ep]['gt_obs']
            gt_obs_1        = SVG_fr[ep]['gt_obs']
            gt_context      = SV2P_01_fr[ep]['contexts']
            gt_context_0    = SV2P_10_fr[ep]['contexts']
            gt_context_1    = SVG_fr[ep]['contexts']
            gt_act          = SV2P_01_fr[ep]['act']
            gt_act_0        = SV2P_10_fr[ep]['act']
            gt_act_1        = SVG_fr[ep]['act']
            pred_sv2p_01_fr = SV2P_01_fr[ep]['pred']
            pred_sv2p_10_fr = SV2P_10_fr[ep]['pred']
            pred_svg_fr     = SVG_fr[ep]['pred']
            # We load multiple versions partly as an extra sanity check.
            assert gt_obs.shape     == gt_obs_0.shape     == gt_obs_1.shape
            assert gt_context.shape == gt_context_0.shape == gt_context_1.shape
            assert gt_act.shape     == gt_act_0.shape     == gt_act_1.shape
            assert np.allclose(gt_act, gt_act_0)
            assert np.allclose(gt_act, gt_act_1)
        elif datatype == 'fabric-01-2021':
            gt_obs_0          = SV2P_10_fnew[ep]['gt_obs']
            gt_obs_1          = SVG_fnew[ep]['gt_obs']
            gt_context_0      = SV2P_10_fnew[ep]['contexts']
            gt_context_1      = SVG_fnew[ep]['contexts']
            gt_act_0          = SV2P_10_fnew[ep]['act']
            gt_act_1          = SVG_fnew[ep]['act']
            pred_sv2p_10_fnew = SV2P_10_fnew[ep]['pred']
            pred_svg_fnew     = SVG_fnew[ep]['pred']
            # We load multiple versions partly as an extra sanity check.
            assert gt_obs_0.shape     == gt_obs_1.shape
            assert gt_context_0.shape == gt_context_1.shape
            assert gt_act_0.shape     == gt_act_1.shape
            assert np.allclose(gt_act_0, gt_act_1)

        # Actually cannot use len(gt_act), that's not the number of actions in an episode
        # Oh, the N can be negative so if that's the case shape of gt_context is (0,)
        num_obs = len(gt_obs_0)
        N = num_obs - HORIZON
        assert N <= gt_context_0.shape[0], f'{N} vs {gt_context_0.shape}'

        # New image for each episode? Actually maybe consider splitting depth vs color?
        # num_steps is number of actions but also need to add padding
        ws = 5                          # whitespace padding
        IM_WIDTH  = (num_obs) * 56 + (num_obs+1) * ws

        # Actually add another column for some SSIM measurements.
        IM_WIDTH += (ws + 56)

        # What do we want in the image? 2 rows for GT color and depth, then 2 for each
        # prediction of SV2P (1 and 10 masks) on the RGB image? N = number of predictions.
        # Then do 1 more row after that just to give empty breathing room to make sure we
        # did this right ... (sanity checks). Update: adding 2 more for depth, and we also
        # have to add the SVG predictions! But don't have 1 mask case for one case.
        if datatype == 'fabric-random':
            nrows = 2 + (N * 6) + 1
        elif datatype == 'fabric-01-2021':
            nrows = 2 + (N * 4) + 1
        nrows = max(3, nrows)  # handle negative N
        IM_HEIGHT = (nrows+1)*ws + (nrows*56)

        # Finally make this PIL Image.
        t_img = Image.new(mode='RGB', size=(IM_WIDTH,IM_HEIGHT), color=(255,255,255))
        draw = ImageDraw.Draw(t_img)

        # Ground truth sequence at the top
        for t in range(num_obs):
            _w = (t * 56) + ((t+1) * ws)
            _h = ws
            frame_rgb = gt_obs_0[t, :, :, :3]
            frame_d   = gt_obs_0[t, :, :, 3:]
            frame_d   = make_depth_img(frame_d)
            t_img.paste(PIL.Image.fromarray(frame_rgb), (_w, _h)          )
            t_img.paste(PIL.Image.fromarray(frame_d),   (_w, _h + 56 + ws))

        # Iterate through all predictions, adding them sequentially. Update: actually
        # we have to check to see if the episode has enough length at all. If an episode
        # has under 5 actions then we could not do prediction on it.
        row = 2
        hoff = 0

        for t in range(0, N):
            # NOTE: we can reorder the way we lay out the rows (e.g., I do SV2P then SVG)
            # if we want, just make sure `row` gets incremented / decremented as needed.
            if datatype == 'fabric-random':
                sv2p_01   = pred_sv2p_01_fr[t,:,:,:,:]  # (H,56,56,4)
                sv2p_10   = pred_sv2p_10_fr[t,:,:,:,:]  # (H,56,56,4)
                svg       = pred_svg_fr[t,:,:,:,:]      # (H,56,56,4)
                context   = gt_context[t,:,:,:]
                context_0 = gt_context_0[t,:,:,:]
                context_1 = gt_context_1[t,:,:,:]
            elif datatype == 'fabric-01-2021':
                sv2p_10   = pred_sv2p_10_fnew[t,:,:,:,:]  # (H,56,56,4)
                svg       = pred_svg_fnew[t,:,:,:,:]      # (H,56,56,4)
                context_0 = gt_context_0[t,:,:,:]
                context_1 = gt_context_1[t,:,:,:]
            # Now we know we can use `sv2p_10` and `svg`, and `sv2p_01` for the old data.

            # Starting width for this group, gets shifted for every batch of predictions.
            _w_start = ws + hoff

            # SV2P ONE MASK. Context, then predicted images.
            if datatype == 'fabric-random':
                img = PIL.Image.fromarray(context[:,:,:3])
                _h = (row+1)*ws + row*56
                t_img.paste(img, (_w_start, _h))
                for h in range(HORIZON):
                    img = PIL.Image.fromarray(sv2p_01[h, :, :, :3])
                    _w = (h+2)*ws + (h+1)*56 + hoff
                    t_img.paste(img, (_w, _h))
                draw_ssim(draw, IM_WIDTH, ws, hoff=_h, groupname='SV2P 01')
                row += 1

            # Next row, SV2P 10 MASK (Edit: putting row += 1 up earlier)
            img = PIL.Image.fromarray(context_0[:,:,:3])
            _h = (row+1)*ws + row*56
            t_img.paste(img, (_w_start, _h))
            for h in range(HORIZON):
                img = PIL.Image.fromarray(sv2p_10[h, :, :, :3])
                _w = (h+2)*ws + (h+1)*56 + hoff
                t_img.paste(img, (_w, _h))
            draw_ssim(draw, IM_WIDTH, ws, hoff=_h, groupname='SV2P 10')

            # Next row, SVG
            row += 1
            img = PIL.Image.fromarray(context_1[:,:,:3])
            _h = (row+1)*ws + row*56
            t_img.paste(img, (_w_start, _h))
            for h in range(HORIZON):
                img = PIL.Image.fromarray(svg[h, :, :, :3])
                _w = (h+2)*ws + (h+1)*56 + hoff
                t_img.paste(img, (_w, _h))
            draw_ssim(draw, IM_WIDTH, ws, hoff=_h, groupname='SVG')

            # SV2P ONE MASK, DEPTH.
            if datatype == 'fabric-random':
                row += 1
                img = PIL.Image.fromarray( make_depth_img(context[:,:,3:]) )
                _h = (row+1)*ws + row*56
                t_img.paste(img, (_w_start, _h))
                for h in range(HORIZON):
                    img = PIL.Image.fromarray( make_depth_img(sv2p_01[h, :, :, 3:]) )
                    _w = (h+2)*ws + (h+1)*56 + hoff
                    t_img.paste(img, (_w, _h))
                draw_ssim(draw, IM_WIDTH, ws, hoff=_h, groupname='SV2P 01')

            # Next row, SV2P 10 MASK, DEPTH
            row += 1
            img = PIL.Image.fromarray( make_depth_img(context_0[:,:,3:]) )
            _h = (row+1)*ws + row*56
            t_img.paste(img, (_w_start, _h))
            for h in range(HORIZON):
                img = PIL.Image.fromarray( make_depth_img(sv2p_10[h, :, :, 3:]) )
                _w = (h+2)*ws + (h+1)*56 + hoff
                t_img.paste(img, (_w, _h))
            draw_ssim(draw, IM_WIDTH, ws, hoff=_h, groupname='SV2P 10')

            # Next row, SVG, DEPTH
            row += 1
            img = PIL.Image.fromarray( make_depth_img(context_1[:,:,3:]) )
            _h = (row+1)*ws + row*56
            t_img.paste(img, (_w_start, _h))
            for h in range(HORIZON):
                img = PIL.Image.fromarray( make_depth_img(svg[h, :, :, 3:]) )
                _w = (h+2)*ws + (h+1)*56 + hoff
                t_img.paste(img, (_w, _h))
            draw_ssim(draw, IM_WIDTH, ws, hoff=_h, groupname='SVG')

            # Prep for next row in next for loop.
            row += 1
            hoff += (ws + 56)

        # Save in BGR format, which means saving with OpenCV.
        estr = str(ep).zfill(3)
        sstr = str(num_obs).zfill(2)
        img_path = f'preds_{datatype}_ep_{estr}_obs_{sstr}_rgbd.png'
        if datatype == 'fabric-random':
            img_path = join('results_fr', img_path)
        else:
            img_path = join('results_fnew', img_path)
        #t_img.save(img_path)  # RGB, but we want BGR, hence do these two:
        t_img_np = np.array(t_img)
        cv2.imwrite(img_path, t_img_np)


if __name__ == "__main__":
    save_images_get_ssim(datatype='fabric-random')
    #save_images_get_ssim(datatype='fabric-01-2021')