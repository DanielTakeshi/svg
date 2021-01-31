# Daniel's Documentation for Stochastic Video Generation

- [Installation](#installation)
- [Stochastic Moving MNIST](#sm-mnist)
- [BAIR Robot Push Data](#bair-data)
- [Fabrics Data](#fabrics-data)
- [Evaluation](#evaluation)

## Installation

The original repository doesn't give information about how to install, so I
mostly followed [this report][4]. For example, we need [an older scikit-image
version][2] to get image metrics to work. I followed [this pull request][1] to
use pytorch=1.0.0 (and presumably torchvision==0.2.1 [since that's the matching
version according to the PyTorch website][3]). All this should be in the conda
`.yml` files to install:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda env create -f env_torch_1.0.yml
conda activate svg
```

Then, create the directory `/data/svg/` which should enable us to run scripts.
If something wrong happens, just restart: `conda env remove -n svg`.

**UPDATE Jan 26, 2021:** I changed this so it's a package. Install with `pip
install -e .`. Also for training please use `SVG.py`.

## SM-MNIST

This data (proposed in the SVG paper itself) is a *stochastic* version of
"moving MNIST." Train SVG-LP:

```
python train_svg_lp.py --dataset smmnist --num_digits 2 --g_dim 128 --z_dim 10 --beta 0.0001 \
        --data_root /data/svg/smmnist --log_dir /data/svg/logs/
```

Note:

- The paper trains models to condition on 5 frames (`--n_past`) and predict the
  next 10 frames (`--n_future`). *Evaluation* uses up to 30 frames (`n_eval`).
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
  sums to 60K items per epoch, seems logical. There is also a train/test split.
- Add `--channels 3` to replicate the input across channels and check if the
  network architecture is compatible. Similarly, can adjust `--image_width`.

It seems to train:

```
~/svg $ python train_svg_lp.py --dataset smmnist --num_digits 2 --g_dim 128 --z_dim 10 --beta 0.0001 --data_root /data/svg/mnist/ --log_dir /data/svg/logs/
/home/seita/miniconda3/envs/svg/lib/python3.7/site-packages/sklearn/externals/joblib/externals/cloudpickle/cloudpickle.py:47: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
  import imp
Random Seed:  1
Namespace(batch_size=100, beta=0.0001, beta1=0.9, channels=1, data_root='/data/svg/mnist/', data_threads=5, dataset='smmnist', epoch_size=600, g_dim=128, image_width=64, last_frame_skip=False, log_dir='/data/svg/logs//smmnist-2/model=dcgan64x64-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=5-n_future=10-lr=0.0020-g_dim=128-z_dim=10-last_frame_skip=False-beta=0.0001000', lr=0.002, model='dcgan', model_dir='', n_eval=30, n_future=10, n_past=5, name='', niter=300, num_digits=2, optimizer='adam', posterior_rnn_layers=1, predictor_rnn_layers=2, prior_rnn_layers=1, rnn_size=256, seed=1, z_dim=10)
N/A% (0 of 600) |                                                                                                                                            | Elapsed Time: 0:00:00 ETA:  --:--:--THCudaCheck FAIL file=/opt/conda/conda-bld/pytorch_1544176307774/work/aten/src/THC/THCGeneral.cpp line=405 error=11 : invalid argument
[00] mse loss: 0.02475 | kld loss: 0.41838 (0)
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
...
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
log dir: /data/svg/logs//smmnist-2/model=dcgan64x64-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=5-n_future=10-lr=0.0020-g_dim=128-z_dim=10-last_frame_skip=False-beta=0.0001000
 72% (437 of 600) |####################################################################################################                                      | Elapsed Time: 0:09:08 ETA:   0:03:25
```

We get `THCudaCheck FAIL` errors, but it seems like training went well?  The
data and log files look something like this:

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
        model.pth
        gen/
            rec_x.png
            sample_x.gif
        plots/
            (empty?)
```

With SM-MNIST, however, the data is generated on-the-fly, according to the
paper, which may explain some slow-downs. Each progress bar that fills up is
one epoch. The `gen` subdirectory has stuff, but `plots` doesn't?


## BAIR Data

TODO -- follow data formatting instructions, test action inclusion?

```
python train_svg_lp.py --dataset bair --model vgg --g_dim 128 --z_dim 64 --beta 0.0001 \
    --n_past 2 --n_future 10 --channels 3 --data_root /path/to/data/ --log_dir /logs/will/be/saved/here/
```


## Fabrics Data

Using the same 3 context frames and 7 future frames training setting from
earlier results, we can train fabric prediction. For evaluation, we shouldn't
use 30 (our trajectories are only of length 15) so maybe use 10? DCGAN is
probably more desirable than VGG here. For most other hyperparameters, I'm
sticking to those used in SM-MNIST, however we have to adjust epoch size and
the iterations. We probably want larger epoch sizes to cut down on disk space
that gets added.

```
python train_svg_lp.py  --dataset fabric-random  --g_dim 128  --z_dim 10  --beta 0.0001 \
    --n_past 3  --n_future 7  --n_eval 10  --channels 4  --image_width 56 \
    --data_root /data/svg/fabric-random  --log_dir /data/svg/logs/ \
    --epoch_size 200  --niter 1500  --action_cond
```

Without `--action_cond`, the model can still generate some predictions, but
actions should make them sharper. Also test the newer dataset which has an
episode length of 10. For this we're defaulting to 2 context and 5 future:

```
python train_svg_lp.py  --dataset fabric-01_2021  --g_dim 128  --z_dim 10  --beta 0.0001 \
    --n_past 2  --n_future 5  --n_eval 7  --channels 4  --image_width 56 \
    --data_root /data/svg/fabric-01_2021  --log_dir /data/svg/logs/ \
    --epoch_size 200  --niter 1500  --action_cond
```

To plot results for MSE and KL Divergence losses, run something like:

```
python plot_svg.py /data/svg/logs/fabric-random/
```

It will iterate through the different models within the subdirectory (the
directories with the very long names) and plot the MSE and KL divergence
losses.


# Evaluation

For SV2P (in other repository) test with these commands inside docker [on this
commit][5]:

```
CUDA_VISIBLE_DEVICES=0 python vismpc/scripts/predict.py \
    --input_img=/data/cloth-visual-mpc/logs/demos-fabric-random-epis_400_COMBINED.pkl \
    --model_dir=/data/pure_random/sv2p_output \
    --data_dir=/data/pure_random/sv2p_data \
    --horizon=5  --batch

CUDA_VISIBLE_DEVICES=0 python vismpc/scripts/predict.py \
    --input_img=/data/cloth-visual-mpc/logs/demos-fabric-random-epis_400_COMBINED.pkl \
    --model_dir=/data/pure_random/sv2p_output_1mask \
    --data_dir=/data/pure_random/sv2p_data_1mask \
    --horizon=5  --batch

CUDA_VISIBLE_DEVICES=0 python vismpc/scripts/predict.py \
    --input_img=/data/cloth-visual-mpc/logs/demos-fabric-01-2021-epis_400_COMBINED.pkl \
    --model_dir=/data/01_2021_backup/sv2p_model_cloth \
    --data_dir=/data/01_2021_backup/sv2p_data_cloth \
    --horizon=5  --batch
```

For SVG, assuming `cloth-visual-mpc` is where all the data are stored, use
(models TBD):

```
python predict_svg_lp.py \
    --model_path=/data/svg/logs-20-Jan-SVG-LP-Mason/fabric-random/model\=dcgan56x56-rnn_size\=256-predictor-posterior-prior-rnn_layers\=2-1-1-n_past\=3-n_future\=7-lr\=0.0020-g_dim\=128-z_dim\=10-last_frame_skip\=False-beta\=0.0001000-act-cond-1/model.pth \
    --data_path=../cloth-visual-mpc/logs/demos-fabric-random-epis_400_COMBINED.pkl

python predict_svg_lp.py \
    --model_path=/data/svg/logs-20-Jan-SVG-LP-Mason/fabric-01_2021/model\=dcgan56x56-rnn_size\=256-predictor-posterior-prior-rnn_layers\=2-1-1-n_past\=2-n_future\=5-lr\=0.0020-g_dim\=128-z_dim\=10-last_frame_skip\=False-beta\=0.0001000-act-cond-1/model.pth \
    --data_path=../cloth-visual-mpc/logs/demos-fabric-01-2021-epis_400_COMBINED.pkl
```

Look at `results_svg/` for the pickle file output, then other results
directories for the IMAGES formed after running `compare_sv2p_svg.py`.  To
measure quality of predictions, look at `compare_sv2p_svg.py`.

**UPDATE**: actually to save in a way that makes it easier to load this later,
please use `svg/SVG.py` script.

I have several predictions:

- [Results][6] for "version 1" of models, which passes in the raw action, and
  saves using the non-recommended Pytorch way.

- [Results][7] for "version 2" of models, which passes in the raw action, and
  saves using the RECOMMENDED way for PyTorch, and uses `svg/SVG.py`.

- [Results][8] for "version 3" of models, which passes in a LEARNED EMBEDDING
  of the raw action (it makes it from 4D to 32D), and saves using the
  RECOMMENDED way for PyTorch, and uses `svg/SVG.py`. There are only a few
  models here, but SVG ends up overfitting badly so we used earlier models.

In all cases, the above results (version 1, version 2, version 3) apply to BOTH
FabricV1 and FabricV2. The "version" is not the version of the fabric dataset,
but the version of the set of neural networks.

With "version 3" and the learned action predictor, here are the SVG parameters:

```
Number of parameters:
  frame p:   1129344
  posterior: 564500
  prior:     572692
  encoder:   3808448
  decoder:   6564676
  act_embed: 1216
total: 12640876
```

[1]:https://github.com/edenton/svg/pull/6
[2]:https://stackoverflow.com/questions/55178229/importerror-cannot-import-name-structural-similarity-error
[3]:https://pytorch.org/get-started/previous-versions/
[4]:https://github.com/edenton/svg/issues/10
[5]:https://github.com/ryanhoque/cloth-visual-mpc/commit/d17e30e7edaa9409c5317a86cb9fb263674b0f65
[6]:https://pastebin.com/raw/rigB93wj
[7]:https://pastebin.com/raw/9GvUAmbp
[8]:https://pastebin.com/raw/fMqSwnXe
