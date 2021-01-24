"""
Compare SV2P and SVG. [SV2P is trained elsewhere, but the predictions file we use
will be in a standard pickle file that we can load anywhere.]

For SV2P prediction format, see:
https://github.com/ryanhoque/cloth-visual-mpc/blob/sv2p/vismpc/scripts/predict.py

NOTE: https://stackoverflow.com/questions/55178229/importerror-cannot-import-name-structural-similarity-error
we are using the older way of doing ssim_metric, so I don't know if documentation is up to date.
Probably easier to do each channel separately, then average, as Dr. Denton does it.

Update Jan 24: iterate through different model snapshots for SVG.
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
np.set_printoptions(precision=3)


pp = argparse.ArgumentParser()
pp.add_argument('--datatype', default='')
pp.add_argument('--svg_model', default='model_0050')
args = pp.parse_args()
assert args.datatype in ['fabric-random', 'fabric-01-2021'], args.datatype


# ----------------------------------------------------------------------------- #
# ------------------------------- DATA FILES ---------------------------------- #
# ----------------------------------------------------------------------------- #
HORIZON = 5
HEAD_SV2P = '/home/seita/cloth-visual-mpc/logs/'
HEAD_SVG  = '/home/seita/svg/results_svg/'

# fr = fabric_random. Always load even though we might be using the other data.
TAIL_SV2P_01_fr = 'demos-fabric-random-epis_400_COMBINED_PREDS_SV2P_01.pkl'
TAIL_SV2P_10_fr = 'demos-fabric-random-epis_400_COMBINED_PREDS_SV2P_10.pkl'
TAIL_SVG_fr    = f'demos-fabric-random-epis_400_COMBINED_PREDS_SVG-LP_{args.svg_model}.pkl'
SV2P_01_fr_pth  = join(HEAD_SV2P, TAIL_SV2P_01_fr)
SV2P_10_fr_pth  = join(HEAD_SV2P, TAIL_SV2P_10_fr)
SVG_fr_pth      = join(HEAD_SVG,  TAIL_SVG_fr)

# fnew = fabric_01-2021 (new data). Always load even though we might be using the other data.
TAIL_SV2P_10_fnew = 'demos-fabric-01-2021-epis_400_COMBINED_PREDS_SV2P_10.pkl'
TAIL_SVG_fnew    = f'demos-fabric-01-2021-epis_400_COMBINED_PREDS_SVG-LP_{args.svg_model}.pkl'
SV2P_10_fnew_pth  = join(HEAD_SV2P, TAIL_SV2P_10_fnew)
SVG_fnew_pth      = join(HEAD_SVG,  TAIL_SVG_fnew)

# These are _lists_, one dict per episode.
with open(SV2P_01_fr_pth, 'rb') as fh:
    SV2P_01_fr = pickle.load(fh)
with open(SV2P_10_fr_pth, 'rb') as fh:
    SV2P_10_fr = pickle.load(fh)
with open(SVG_fr_pth, 'rb') as fh:
    SVG_fr     = pickle.load(fh)

# fnew data
with open(SV2P_10_fnew_pth, 'rb') as fh:
    SV2P_10_fnew = pickle.load(fh)
with open(SVG_fnew_pth, 'rb') as fh:
    SVG_fnew     = pickle.load(fh)

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


def draw_ssim(draw, IM_WIDTH, ws, hoff, groupname, ssims):
    # Draw some structural similarity values as a sanity check.
    fontsize = 10
    ssim_width = IM_WIDTH - (ws + 56)  # Make this LAST column
    text_0 = f'ssim {ssims[0]:0.2f}'
    text_1 = f'ssim {ssims[1]:0.2f}'
    text_2 = f'ssim {ssims[2]:0.2f}'
    text_3 = f'ssim {ssims[3]:0.2f}'
    text_4 = f'ssim {ssims[4]:0.2f}'
    off = 9
    if 'SVG' in groupname:
        groupname += args.svg_model
        groupname = groupname.replace('model_', '')
    draw.text((ssim_width, hoff+0*off), groupname, fill=(0,0,0,0))
    draw.text((ssim_width, hoff+1*off), text_0, fill=(0,0,0,0))
    draw.text((ssim_width, hoff+2*off), text_1, fill=(0,0,0,0))
    draw.text((ssim_width, hoff+3*off), text_2, fill=(0,0,0,0))
    draw.text((ssim_width, hoff+4*off), text_3, fill=(0,0,0,0))
    draw.text((ssim_width, hoff+5*off), text_4, fill=(0,0,0,0))


def eval_ssim(true_imgs, pred_imgs, channel='both'):
    # By default, sum SSIMs across all channels, and average.
    assert channel in ['color', 'depth', 'both'], channel
    assert true_imgs.shape == pred_imgs.shape == (HORIZON, 56, 56, 4), true_imgs.shape
    if channel == 'both':
        start, end = 0, 4
    elif channel == 'color':
        start, end = 0, 3
    elif channel == 'depth':
        start, end = 3, 4

    ssims = []
    for t in range(HORIZON):
        true_img = true_imgs[t]
        pred_img = pred_imgs[t]
        ssim = 0
        for c in range(start, end):
            X = true_img[:,:,c]
            Y = pred_img[:,:,c]
            if channel == 'depth':
                # Makes it 3 channel, but we only want the 1st (it's replicated).
                X = (make_depth_img(X))[:,:,0]
                Y = (make_depth_img(Y))[:,:,0]
            ssim += ssim_metric(X=X, Y=Y)
        ssim /= (end-start)
        assert -1 <= ssim <= 1, ssim
        ssims.append(ssim)
    return ssims


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
    nb_eps = len(SV2P_10_fr)

    # SSIM: dict, maps model type (e.g., 'svg') --> ssim matrix of dim (num_eps, HORIZON)
    # We can then average across columns to get SSIM for 1-ahead predictions, 2-ahead, etc.
    # Imagine these as a grid of (episodes) x (horizon). The item in the 'top left' is for
    # 1st episode, 1-ahead prediction. In that episode, we have `N` sets of predictions, so
    # these store the average of the k-ahead predictions, k in {1,2,3,4,5=HORIZON}. BUT,
    # because some episodes are too short, it's easier to append a bunch of HORIZON-length
    # lists and then call np.array() on it later.
    SSIM_C = {'sv2p_01': [], 'sv2p_10': [], 'svg': [],}
    SSIM_D = {'sv2p_01': [], 'sv2p_10': [], 'svg': [],}

    for ep in range(nb_eps):
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

        # Iterate through all PREDICTIONS, adding them sequentially. Update: actually
        # we have to check to see if the episode has enough length at all. If an episode
        # has under 5 actions then we could not do prediction on it.
        row = 2
        hoff = 0

        # SSIM FOR THIS EPISODE ONLY.
        ep_ssims_sv2p_01_c = []
        ep_ssims_sv2p_10_c = []
        ep_ssims_svg_c     = []
        ep_ssims_sv2p_01_d = []
        ep_ssims_sv2p_10_d = []
        ep_ssims_svg_d     = []

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
            # Here is also the ground truth for this full set of predictions.
            true_imgs = gt_obs_0[t:t+HORIZON, :, :, :]
            assert true_imgs.shape == sv2p_10.shape == svg.shape, true_imgs.shape

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
                ssims = eval_ssim(true_imgs=true_imgs, pred_imgs=sv2p_01, channel='color')
                draw_ssim(draw, IM_WIDTH, ws, hoff=_h, groupname='SV2P 01', ssims=ssims)
                ep_ssims_sv2p_01_c.append(ssims)
                row += 1

            # Next row, SV2P 10 MASK (Edit: putting row += 1 up earlier)
            img = PIL.Image.fromarray(context_0[:,:,:3])
            _h = (row+1)*ws + row*56
            t_img.paste(img, (_w_start, _h))
            for h in range(HORIZON):
                img = PIL.Image.fromarray(sv2p_10[h, :, :, :3])
                _w = (h+2)*ws + (h+1)*56 + hoff
                t_img.paste(img, (_w, _h))
            ssims = eval_ssim(true_imgs=true_imgs, pred_imgs=sv2p_10, channel='color')
            draw_ssim(draw, IM_WIDTH, ws, hoff=_h, groupname='SV2P 10', ssims=ssims)
            ep_ssims_sv2p_10_c.append(ssims)

            # Next row, SVG
            row += 1
            img = PIL.Image.fromarray(context_1[:,:,:3])
            _h = (row+1)*ws + row*56
            t_img.paste(img, (_w_start, _h))
            for h in range(HORIZON):
                img = PIL.Image.fromarray(svg[h, :, :, :3])
                _w = (h+2)*ws + (h+1)*56 + hoff
                t_img.paste(img, (_w, _h))
            ssims = eval_ssim(true_imgs=true_imgs, pred_imgs=svg, channel='color')
            draw_ssim(draw, IM_WIDTH, ws, hoff=_h, groupname='SVG', ssims=ssims)
            ep_ssims_svg_c.append(ssims)

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
                ssims = eval_ssim(true_imgs=true_imgs, pred_imgs=sv2p_01, channel='depth')
                draw_ssim(draw, IM_WIDTH, ws, hoff=_h, groupname='SV2P 01', ssims=ssims)
                ep_ssims_sv2p_01_d.append(ssims)

            # Next row, SV2P 10 MASK, DEPTH
            row += 1
            img = PIL.Image.fromarray( make_depth_img(context_0[:,:,3:]) )
            _h = (row+1)*ws + row*56
            t_img.paste(img, (_w_start, _h))
            for h in range(HORIZON):
                img = PIL.Image.fromarray( make_depth_img(sv2p_10[h, :, :, 3:]) )
                _w = (h+2)*ws + (h+1)*56 + hoff
                t_img.paste(img, (_w, _h))
            ssims = eval_ssim(true_imgs=true_imgs, pred_imgs=sv2p_10, channel='depth')
            draw_ssim(draw, IM_WIDTH, ws, hoff=_h, groupname='SV2P 10', ssims=ssims)
            ep_ssims_sv2p_10_d.append(ssims)

            # Next row, SVG, DEPTH
            row += 1
            img = PIL.Image.fromarray( make_depth_img(context_1[:,:,3:]) )
            _h = (row+1)*ws + row*56
            t_img.paste(img, (_w_start, _h))
            for h in range(HORIZON):
                img = PIL.Image.fromarray( make_depth_img(svg[h, :, :, 3:]) )
                _w = (h+2)*ws + (h+1)*56 + hoff
                t_img.paste(img, (_w, _h))
            ssims = eval_ssim(true_imgs=true_imgs, pred_imgs=svg, channel='depth')
            draw_ssim(draw, IM_WIDTH, ws, hoff=_h, groupname='SVG', ssims=ssims)
            ep_ssims_svg_d.append(ssims)

            # Prep for next row in next for loop.
            row += 1
            hoff += (ws + 56)

        # SSIMs if at least 1 prediction. Computed {1-ahead, ..., H-ahead} SSIMs for this episode.
        if N >= 1:
            ep_ssims_sv2p_01_c = np.array( ep_ssims_sv2p_01_c )
            ep_ssims_sv2p_10_c = np.array( ep_ssims_sv2p_10_c )
            ep_ssims_svg_c     = np.array( ep_ssims_svg_c     )
            ep_ssims_sv2p_01_d = np.array( ep_ssims_sv2p_01_d )
            ep_ssims_sv2p_10_d = np.array( ep_ssims_sv2p_10_d )
            ep_ssims_svg_d     = np.array( ep_ssims_svg_d     )
            assert ep_ssims_svg_c.shape == (num_obs - HORIZON, HORIZON), \
                    f'{num_obs}, {ep_ssims_svg_c.shape}'

            if datatype == 'fabric-random':
                ep_ssims_sv2p_01_c = np.mean(ep_ssims_sv2p_01_c, axis=0)
                ep_ssims_sv2p_01_d = np.mean(ep_ssims_sv2p_01_d, axis=0)
            ep_ssims_sv2p_10_c = np.mean(ep_ssims_sv2p_10_c, axis=0)
            ep_ssims_svg_c     = np.mean(ep_ssims_svg_c    , axis=0)
            ep_ssims_sv2p_10_d = np.mean(ep_ssims_sv2p_10_d, axis=0)
            ep_ssims_svg_d     = np.mean(ep_ssims_svg_d    , axis=0)

            SSIM_C['sv2p_01'].append( ep_ssims_sv2p_01_c )
            SSIM_C['sv2p_10'].append( ep_ssims_sv2p_10_c )
            SSIM_C['svg'    ].append( ep_ssims_svg_c     )
            SSIM_D['sv2p_01'].append( ep_ssims_sv2p_01_d )
            SSIM_D['sv2p_10'].append( ep_ssims_sv2p_10_d )
            SSIM_D['svg'    ].append( ep_ssims_svg_d     )

        # Save in BGR format, which means saving with OpenCV.
        estr = str(ep).zfill(3)
        sstr = str(num_obs).zfill(2)
        img_path = f'preds_{datatype}_ep_{estr}_obs_{sstr}_rgbd.png'
        if datatype == 'fabric-random':
            head_dir = join('results_frand', args.svg_model)
            os.makedirs(head_dir, exist_ok=True)
            img_path = join(head_dir, img_path)
        else:
            head_dir = join('results_fnew', args.svg_model)
            os.makedirs(head_dir, exist_ok=True)
            img_path = join(head_dir, img_path)
        #t_img.save(img_path)  # RGB, but we want BGR, hence do these two:
        t_img_np = np.array(t_img)
        cv2.imwrite(img_path, t_img_np)

    # Report SSIM metrics
    print(f'\nSome SSIM metrics for: {args}:\n')
    print('Length of lists: {} <= episodes {}'.format(len(SSIM_C['sv2p_10']), nb_eps))

    # Form numpy arrays.
    if datatype == 'fabric-random':
        SSIM_C['sv2p_01'] = np.array(SSIM_C['sv2p_01'])
        SSIM_D['sv2p_01'] = np.array(SSIM_D['sv2p_01'])
    SSIM_C['sv2p_10'] = np.array(SSIM_C['sv2p_10'])
    SSIM_C['svg'    ] = np.array(SSIM_C['svg'    ])
    SSIM_D['sv2p_10'] = np.array(SSIM_D['sv2p_10'])
    SSIM_D['svg'    ] = np.array(SSIM_D['svg'    ])

    # Print values. To make things easier to copy and paste, do color then depth.
    if datatype == 'fabric-random':
        print('SV2P01, C. {}'.format( np.mean(SSIM_C['sv2p_01'], axis=0)) )
    print('SV2P10, C. {}'.format( np.mean(SSIM_C['sv2p_10'], axis=0)) )
    print('SVG, C.    {}'.format( np.mean(SSIM_C['svg'],     axis=0)) )
    if datatype == 'fabric-random':
        print('SV2P01, D. {}'.format( np.mean(SSIM_D['sv2p_01'], axis=0)) )
    print('SV2P10, D. {}'.format( np.mean(SSIM_D['sv2p_10'], axis=0)) )
    print('SVG, D.    {}'.format( np.mean(SSIM_D['svg'],     axis=0)) )
    print('\nNote, shape SV2P, SVG: {}, {}'.format(SSIM_C['sv2p_10'].shape, SSIM_C['svg'].shape))

    print('\nNow standard deviations among the episodes:\n')
    if datatype == 'fabric-random':
        print('SV2P01, C. {}'.format( np.std(SSIM_C['sv2p_01'], axis=0)) )
    print('SV2P10, C. {}'.format( np.std(SSIM_C['sv2p_10'], axis=0)) )
    print('SVG, C.    {}'.format( np.std(SSIM_C['svg'],     axis=0)) )
    if datatype == 'fabric-random':
        print('SV2P01, D. {}'.format( np.std(SSIM_D['sv2p_01'], axis=0)) )
    print('SV2P10, D. {}'.format( np.std(SSIM_D['sv2p_10'], axis=0)) )
    print('SVG, D.    {}'.format( np.std(SSIM_D['svg'],     axis=0)) )
    print()


if __name__ == "__main__":
    save_images_get_ssim(datatype=args.datatype)