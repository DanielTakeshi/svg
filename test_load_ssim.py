"""This should be the same as `predict_svg_lp.py` except we load using PyTorch's recommended way.

Please us this going forward, as of Jan 26, 2021.
"""
import argparse
import os
import sys
import pickle
import random
import numpy as np
#import SVG
from svg.SVG import SVG


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--n_future',  default=5)
    parser.add_argument('--data_path', default='', help='not used in SVG, just for this script')
    opt = parser.parse_args()
    svg_model = SVG(opt)
    print('\nSuccessfully loaded! Woo hoo, see the opt above. Now try prediction.')

    # Somewhat hacky and assumes a specific directory structure. Update: making predictions
    # for a variety of models, hence let's put the model path at the end.
    _, tail = os.path.split(opt.model_dir)
    tail = tail.replace('.pth', '')
    assert 'cloth-visual-mpc/logs/' in opt.data_path, opt.data_path
    outname = (opt.data_path).replace('cloth-visual-mpc/logs/', 'svg/results_svg_2/')
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
            currim = episode['obs'][i]                      # shape (56, 56, 4), type uint8
            curracts = all_acts[i:i + opt.n_future]         # shape (H, 4)
            pred = svg_model.predict(x=currim, x_acts=curracts)   # shape (H, 56, 56, 4), float32, in (0,1)
            preds.append(pred)                              # ALL preds: \hat{x}_{i+1 : i+H}
            acts.append(curracts)                           # input actions: a_{i : i+H-1}
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
