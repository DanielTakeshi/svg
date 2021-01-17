"""Plot the results.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
# https://stackoverflow.com/questions/43080259/
# no-outlines-on-bins-of-matplotlib-histograms-or-seaborn-distplots/43080772
plt.rcParams["patch.force_edgecolor"] = True

import os
import sys
import random
import argparse
import numpy as np
import pickle
import utils
from os.path import join

# Matplotlib stuff
titlesize = 32
xsize = 30
ysize = 30
ticksize = 28
legendsize = 24
er_alpha = 0.25
lw = 3


def plot_losses(args, directory):
    """Plot losses from the saved pickle file."""
    nrows, ncols = 1, 2
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=(11*ncols, 8*nrows))

    pkl_file = join(directory, 'losses.pkl')
    assert os.path.exists(pkl_file), pkl_file
    with open(pkl_file, 'rb') as fh:
        data = pickle.load(fh)

    xs = data['epoch']
    cumsum = data['tot_train'][-1]
    print(f'Plotting over {len(xs)} epochs, total number of training samples consumed: {cumsum}')
    ax[0,0].plot(xs, data['mse_loss'], lw=lw)
    ax[0,1].plot(xs, data['kld_loss'], lw=lw)

    # Bells and whistles.
    ax[0,0].set_title(f'MSE Loss', size=titlesize)
    ax[0,1].set_title(f'KL-Div Loss', size=titlesize)
    ax[0,0].set_xlabel('Train Epochs', size=xsize)
    ax[0,1].set_xlabel('Train Epochs', size=xsize)
    ax[0,0].set_ylabel('Loss (MSE)', size=ysize)
    ax[0,1].set_ylabel('Loss (KL Div)', size=ysize)
    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
    plt.tight_layout()

    # Save .png in the same log directory.
    fig_suffix = 'plot_losses.png'.format()
    figname = join(directory, fig_suffix)
    plt.savefig(figname)
    print("Just saved:\n\t{}\n".format(figname))


if __name__ == "__main__":
    # Note: actually path should be one level BEFORE the very long 'model=[...]' phrasing
    # otherwise it'd be a pain to type this in.
    pp = argparse.ArgumentParser()
    pp.add_argument('--path', type=str, default='Path to stored file')
    args = pp.parse_args()
    assert os.path.exists(args.path), args.path

    # Let's extract data directories.
    dirs = sorted([join(args.path, x) for x in os.listdir(args.path)])
    print(f'\nNow plotting over {len(dirs)} directories.')
    for d in dirs:
        plot_losses(args, directory=d)