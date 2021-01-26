"""Daniel: should use this to load data and evaluate.
Let's leave the generate_svg_lp.py script alone in case we want to use it later.
Setting defaults to n_past=1 and n_future=5 to be consistent with RSS 2020 and SV2P.

And also for the sake of visualizations + SSIM, we're just going to generate one
set of predictions (nsample=1), which is again similar to what we did in SV2P.
I am also assuming it's action-conditioned by default and act_dim=4. EXACT CALLS:
see the bash script `predict_svg.sh`.
"""
import os
import sys
import pickle
import random
import argparse
import itertools
import progressbar
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from scipy.ndimage.filters import gaussian_filter
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--seed',       default=1, type=int, help='manual seed')
parser.add_argument('--batch_size', default=1, type=int, help='batch size, definitely easier with 1')
parser.add_argument('--model_path', default='', help='path to model')
parser.add_argument('--data_path',  default='', help='path to prediction data')
parser.add_argument('--log_dir',    default='results_svg/', help='directory to save generations to')
parser.add_argument('--n_past',     default=1, type=int, help='number of frames to condition on')
parser.add_argument('--n_future',   default=5, type=int, help='number of frames to predict')
parser.add_argument('--nsample',    default=1, type=int, help='number of samples')
parser.add_argument('--act_dim',    default=4, type=int, help='Need to adjust architecture a bit')
opt = parser.parse_args()

os.makedirs('%s' % opt.log_dir, exist_ok=True)
opt.n_eval = opt.n_past + opt.n_future
opt.max_step = opt.n_eval
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

# Load models using the "non-recommended" way but seems to work in SpinningUp ...
tmp = torch.load(opt.model_path)
frame_predictor = tmp['frame_predictor']
posterior = tmp['posterior']
prior = tmp['prior']
frame_predictor.eval()
prior.eval()
posterior.eval()
encoder = tmp['encoder']
decoder = tmp['decoder']
encoder.eval()  # TODO(daniel) why was this originally train mode? we're evaluating?
decoder.eval()  # TODO(daniel) why was this originally train mode? we're evaluating?
frame_predictor.batch_size = opt.batch_size
posterior.batch_size = opt.batch_size
prior.batch_size = opt.batch_size
opt.g_dim = tmp['opt'].g_dim
opt.z_dim = tmp['opt'].z_dim
opt.num_digits = tmp['opt'].num_digits

# Transfer to GPU, set options.
frame_predictor.cuda()
posterior.cuda()
prior.cuda()
encoder.cuda()
decoder.cuda()

opt.dataset = tmp['opt'].dataset
opt.last_frame_skip = tmp['opt'].last_frame_skip
opt.channels = tmp['opt'].channels
opt.image_width = tmp['opt'].image_width
print(opt)


def predict(x, x_acts):
    """Daniel: making this a prediction method, like SV2P's prediction.

    Since n_past=1, we definitely need `i <= opt.n_past` to get ANY skip connection.
    Daniel: as usual, iter `i` means x_{i-1} is most recently conditioned frame, SVG
    SAMPLES z_i and x_i. If it's before n_past, just use ground truth x_i for image.
    Code mirrors the train_svg_lp's `plot()` function.

    Need to do some reshaping. Input `x` has shape (56,56,4) but want (1,4,56,56), so move
    the channel to leading dimension, then add np.newaxis. Then convert to PyTorch. That
    should be x_in; more generally we don't really need `x` to be a list since the context
    frames should just be 1 -- check if that's not the case. See data/fabrics.py and the
    utils.normalize_data for correct shapes.

    AH! Just remembered that we have to get the range into [0,1].
    """
    assert opt.n_past == 1
    gen_seq = []
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    prior.hidden = prior.init_hidden()

    # Reshape single context frame. If we actually do more than 1 context frame, we should
    # instead format `x` like we normally do during training by having it as a list, etc.
    x_in = x.transpose(2,0,1)[np.newaxis,:]
    x_in = np.array(x_in).astype(np.float32) / 255.0     # Only major processing was div by 255.
    x_in = torch.from_numpy(x_in).cuda()

    for i in range(1, opt.n_eval):
        h = encoder(x_in)
        if opt.last_frame_skip or i <= opt.n_past:
            h, skip = h
        else:
            h, _ = h

        # Daniel: action conditioning. Updating h: (B,g_dim) --> (B, g_dim+4). B=1 here.
        # Given a_{i-1} we use that with Enc(prior_image) to get (predicted) x_i.
        if x_acts is not None:
            x_a = (x_acts[i-1])[np.newaxis,:]
            x_a = np.array(x_a).astype(np.float32)
            x_a = torch.from_numpy(x_a).cuda()
            h = torch.cat([h, x_a], dim=1)  # overrides h

        h = h.detach()
        if i < opt.n_past:
            h_target = encoder(x[i])
            h_target = h_target[0].detach()
            z_t, _, _ = posterior(h_target)
            prior(h)
            frame_predictor(torch.cat([h, z_t], 1))
            x_in = x[i]
        else:
            z_t, _, _ = prior(h)
            h = frame_predictor(torch.cat([h, z_t], 1)).detach()
            x_in = decoder([h, skip]).detach()
            gen_seq.append(x_in.data.cpu().numpy())

    # Reshape it to match desired output for consistency with SV2P.
    prediction = np.array(gen_seq)              # (n_future, 1, 4, 56, 56)
    prediction = np.squeeze(prediction)         # (n_future, 4, 56, 56)
    prediction = prediction.transpose(0,2,3,1)  # (n_future, 56, 56, 4)
    return prediction

# ---------------------------------------------------------------------------------------------------- #
# Daniel: the data should be loaded using something similar to:
# https://github.com/ryanhoque/cloth-visual-mpc/blob/sv2p/vismpc/scripts/predict.py
# Pkl's entries have: dict_keys(['obs', 'act', 'info', 'rew', 'done'])
# 'obs' needs a `numpy.array(obs)` call to make it shape (steps+1, 56, 56, 4) then 'act'
# needs `numpy.array(act)` to make it shape (steps, 4), it's originally a list of 4-tuples.
# ---------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Somewhat hacky and assumes a specific directory structure. Update: making predictions
    # for a variety of models, hence let's put the model path at the end.
    _, tail = os.path.split(opt.model_path)
    tail = tail.replace('.pth', '')
    assert 'cloth-visual-mpc/logs/' in opt.data_path, opt.data_path
    outname = (opt.data_path).replace('cloth-visual-mpc/logs/', 'svg/results_svg_1/')
    outname = (outname).replace('.pkl', f'_PREDS_SVG-LP_{tail}.pkl')

    # Load, usually from the cloth-visual-mpc/logs directory.
    with open(opt.data_path, 'rb') as fh:
        pkl = pickle.load(fh)
    output = list()

    for episode in pkl:
        num_steps = len(episode['act'])
        all_acts = np.array(episode['act'])
        preds = []
        acts = []
        contexts = []
        for i in range(num_steps + 1 - opt.n_future):
            currim = episode['obs'][i]                 # shape (56, 56, 4), type uint8
            curracts = all_acts[i:i + opt.n_future]    # shape (H, 4)
            pred = predict(x=currim, x_acts=curracts)  # shape (H, 56, 56, 4), float32, in (0,1)
            preds.append(pred)                         # ALL preds: \hat{x}_{i+1 : i+H}
            acts.append(curracts)                      # input actions: a_{i : i+H-1}
            contexts.append(currim)
        # With my way of saving, shapes are (N, H, 56, 56, 4) and (N, H, 4)
        # where N=num_steps-H+1. (All predictions given this episode + horizon)
        gt_obs_all = np.array(episode['obs']).astype(np.uint8)
        contexts_all = np.array(contexts).astype(np.uint8)
        preds_all = np.array(preds) * 255.0     # NOTE: multiply by 255 since output is all in (0,1).
        preds_all = preds_all.astype(np.uint8)
        curr_output = {'pred': preds_all,
                       'act': np.array(acts),
                       'gt_obs': gt_obs_all,
                       'contexts': contexts_all}
        output.append(curr_output)
        if len(output) % 20 == 0:
            print('finished {} episodes ...'.format(len(output)))

    with open(outname, 'wb') as fh:
        pickle.dump(output, fh)
    print(f'Look for output: {outname}')