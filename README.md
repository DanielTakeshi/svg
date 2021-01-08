# Stochastic Video Generation with a Learned Prior

This is code for the paper [Stochastic Video Generation with a Learned Prior](https://arxiv.org/abs/1802.07687) by Emily Denton and Rob Fergus. See the [project page](https://sites.google.com/view/svglp/) for details and generated video sequences.

##  Training on Stochastic Moving MNIST (SM-MNIST)
To train the SVG-LP model on the 2 digit SM-MNIST dataset run:
```
python train_svg_lp.py --dataset smmnist --num_digits 2 --g_dim 128 --z_dim 10 --beta 0.0001 --data_root /path/to/data/ --log_dir /logs/will/be/saved/here/
```
If the MNIST dataset doesn't exist, it will be downloaded to the specified path.

## BAIR robot push dataset
To download the BAIR robot push dataset run:
```
sh data/download_bair.sh /path/to/data/
```
This will download the dataset in tfrecord format into the specified directory. To train the pytorch models, we need to first convert the tfrecord data into .png images by running:
```
python data/convert_bair.py --data_dir /path/to/data/
```
This may take some time. Images will be saved in ```/path/to/data/processeddata```.
Now we can train the SVG-LP model by running:
```
python train_svg_lp.py --dataset bair --model vgg --g_dim 128 --z_dim 64 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --data_root /path/to/data/ --log_dir /logs/will/be/saved/here/
```

To generate images with a pretrained SVG-LP model run:
```
python generate_svg_lp.py --model_path pretrained_models/svglp_bair.pth --log_dir /generated/images/will/save/here/
```


## KTH action dataset
First download the KTH action recognition dataset by running:
```
sh data/download_kth.sh /my/kth/data/path/
```
where /my/kth/data/path/ is the directory the data will be downloaded into. Next, convert the downloaded .avi files into .png's for the data loader. To do this you'll want [ffmpeg](https://ffmpeg.org/) installed. The following script will do the conversion, but beware, it's written in lua (sorry!):
```
th data/convert_kth.lua --dataRoot /my/kth/data/path/ --imageSize 64
```
The ```--imageSize``` flag specifiec the image resolution. Experimental results in the paper used 128x128, but you can also train a model on 64x64 and it will train much faster.
To train the SVG-FP model on 64x64 KTH videos run:
```
python train_svg_fp.py --dataset kth --image_width  64 --model vgg --g_dim 128 --z_dim 24 --beta 0.000001 --n_past 10 --n_future 10 --channels 1 --lr 0.0008 --data_root /path/to/data/ --log_dir /logs/will/be/saved/here/
```

# Daniel's Documentation

- [Installation](#installation)
- [Stochastic Moving MNIST](#sm-mnist)
- [BAIR Robot Push Data](#bair-data)
- [Fabrics Data](#fabrics-data)

## Installation

The original repository doesn't give information about how to install, so
here's what I had to do, plus a few other comments:

- Mostly follow [this issue report][4] to match versions. For example, we need
  [an older scikit-image version][2] to get image metrics to work.
- I followed [this pull request][1] to use pytorch=1.0.0 (and presumably
  torchvision==0.2.1 [since that's the matching version according to the
  PyTorch website][3]).
- TensorFlow is only used for some transformations, and not for training.

I put this information into conda .yml files to install:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda env create -f env_torch_1.0.yml
conda activate svg
```

Then, create the directory `/data/svg/` which should enable us to run scripts.

If something wrong happens, just restart. :)

```
conda env remove -n svg
```

## SM-MNIST

This data (proposed in the SVG paper itself) is a *stochastic* version of
"moving MNIST." Train SVG-LP:

```
python train_svg_lp.py --dataset smmnist --num_digits 2 --g_dim 128 --z_dim 10 --beta 0.0001 \
        --data_root /data/svg/smmnist --log_dir /data/svg/logs/
```

Note:

- The paper tests with 1 or 2 moving digits (though I only see 2). Train models
  to condition on 5 frames (`--n_past`) and predict the next 10 frames
  (`--n_future`). *Evaluation* uses 30 frames (`n_eval`).
- Uses a DCGAN model architecture for the *frame encoder*, the default for `--model`.
- `|g|=128`, size of encoder output (so we go from image --> vector of this
  size), and decoder input (vector of this size --> image)
- `|z|=10`, the dimension of `z_t`, the latent variable at time `t`, so it's
  sampled from a 10-D Gaussian prior.
- When training, we can track qualitative progress (see `smnist-2/model-name/gen/`).
- `--beta` is for the KL loss, `--beta1` (not shown here) is for the Adam
  optimizer's momentum.
- For this and other datasets, LSTMs use 256 cells in each layer.
- Defaults to 300 epochs, with 600 batches per epoch, batch size is 100, which
  sums to 60K items per epoch. Seems logical but I asusme there's a train/test
  split.

It seems to train:

```
(svg) seita@starship:~/svg (master) $ python train_svg_lp.py --dataset smmnist --num_digits 2 --g_dim 128 --z_dim 10 --beta 0.0001 --data_root /data/svg/mnist/ --log_dir /data/svg/logs/
/home/seita/miniconda3/envs/svg/lib/python3.7/site-packages/sklearn/externals/joblib/externals/cloudpickle/cloudpickle.py:47: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
  import imp
Random Seed:  1
Namespace(batch_size=100, beta=0.0001, beta1=0.9, channels=1, data_root='/data/svg/mnist/', data_threads=5, dataset='smmnist', epoch_size=600, g_dim=128, image_width=64, last_frame_skip=False, log_dir='/data/svg/logs//smmnist-2/model=dcgan64x64-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=5-n_future=10-lr=0.0020-g_dim=128-z_dim=10-last_frame_skip=False-beta=0.0001000', lr=0.002, model='dcgan', model_dir='', n_eval=30, n_future=10, n_past=5, name='', niter=300, num_digits=2, optimizer='adam', posterior_rnn_layers=1, predictor_rnn_layers=2, prior_rnn_layers=1, rnn_size=256, seed=1, z_dim=10)
N/A% (0 of 600) |                                                                                                                                            | Elapsed Time: 0:00:00 ETA:  --:--:--THCudaCheck FAIL file=/opt/conda/conda-bld/pytorch_1544176307774/work/aten/src/THC/THCGeneral.cpp line=405 error=11 : invalid argument
[00] mse loss: 0.02475 | kld loss: 0.41838 (0)
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
log dir: /data/svg/logs//smmnist-2/model=dcgan64x64-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=5-n_future=10-lr=0.0020-g_dim=128-z_dim=10-last_frame_skip=False-beta=0.0001000
 72% (437 of 600) |####################################################################################################                                      | Elapsed Time: 0:09:08 ETA:   0:03:25
```

However, we get `THCudaCheck FAIL` errors?

The data and log files look something like this:

```
/data/svg/mnist/
    processed/
        test.pt
        training.pt
    raw/
        t10k-images-idx3-ubyte
        t10k-labels-idx1-ubyte
        train-images-idx3-ubyte
        train-labels-idx1-ubyte
/data/svg/logs/smmnist-2/
    'model=dcgan64x64-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=5-n_future=10-lr=0.0020-g_dim=128-z_dim=10-last_frame_skip=False-beta=0.0001000'
```

With SM-MNIST, however, the data is generated on-the-fly, according to the
paper, which may explain some slow-downs. Each progress bar that fills up is
one epoch.


## BAIR Data

TODO



## Fabrics Data

TODO

[1]:https://github.com/edenton/svg/pull/6
[2]:https://stackoverflow.com/questions/55178229/importerror-cannot-import-name-structural-similarity-error
[3]:https://pytorch.org/get-started/previous-versions/
[4]:https://github.com/edenton/svg/issues/10
