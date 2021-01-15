"""
SVG with LP. There are 5 models, instead of 4 as in FP, because we add `prior` as a model.
B = batch size.

Decoder: (B, images) --> (B, g_dim)
Encoder: (B, g_dim) --> (B, images)

As I know by now, both have skip connections to enable copying the prior frame.

Prior (Gaussian LSTM):      p_psi(z_t | x_{1:t-1})
Posterior (Gaussian LSTM):  q_phi(z_t | x_{1:t})
Frame Predictor (LSTM):     p_theta(x_t | x_{1:t-1}, z_{1:t})

The paper mentions a prediction model a prior distribution, and an inference network.
The prediction model combines the encoder, decoder, and the frame predictor.
The posterior must be the inference network because the KL divergence loss term
tries to keep the outputs (mu, logstd) of both `p_psi` and `q_phi` close.

How does training work? We iterate through n_past+n_future frames, and call the
encoder. Recall that the encoder returns both the smaller vector AND the intermediate
outputs, so that we can run skip connections to the decoder to reconstruct the images.
REMEMBER: prior, posterior, frame predictor take in x_t notationally, but IN PRACTICE
they take in ENCODED images, hence I like writing E[x_t] or E[x_{1:t}], etc. (It's not
an expectation, don't get confused.)

Whew, see docs below, I think I get it. Any remaining confusions:
    TODO(daniel) The paper says the inference (i.e., posterior) network is not used
    at test time. But it's clearly being used in both plot methods.
"""
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
import progressbar
import numpy as np
import sys
import pickle
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/lp', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--dataset', default='smmnist', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict during training')
parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict during eval')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=10, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
parser.add_argument('--action_cond', action='store_true', default=False, help='true to make this action conditioned')
parser.add_argument('--act_design', type=int, default=1, help='TODO(daniel)')
opt = parser.parse_args()

if opt.model_dir != '':
    # load model and continue training from checkpoint
    saved_model = torch.load('%s/model.pth' % opt.model_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.log_dir = '%s/continued' % opt.log_dir
else:
    name = 'model=%s%dx%d-rnn_size=%d-predictor-posterior-prior-rnn_layers=%d-%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f%s' % (
            opt.model, opt.image_width, opt.image_width, opt.rnn_size, opt.predictor_rnn_layers,
            opt.posterior_rnn_layers, opt.prior_rnn_layers, opt.n_past, opt.n_future, opt.lr, opt.g_dim,
            opt.z_dim, opt.last_frame_skip, opt.beta, opt.name)
    if opt.action_cond:
        name += '-act-cond-{}'.format(opt.act_design)
    if opt.dataset == 'smmnist':
        opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
    else:
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

# ---------------- load the models  ----------------
print(opt)

# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = optim.Adam
elif opt.optimizer == 'rmsprop':
    opt.optimizer = optim.RMSprop
elif opt.optimizer == 'sgd':
    opt.optimizer = optim.SGD
else:
    raise ValueError('Unknown optimizer: %s' % opt.optimizer)

# --------------------------------------- MODELS --------------------------------------------- #
import models.lstm as lstm_models
if opt.model_dir != '':
    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
    prior = saved_model['prior']
else:
    frame_predictor = lstm_models.lstm(opt.g_dim+opt.z_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
    posterior       = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
    prior           = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers,     opt.batch_size)
    frame_predictor.apply(utils.init_weights)
    posterior.apply(utils.init_weights)
    prior.apply(utils.init_weights)

if opt.model == 'dcgan':
    if opt.image_width == 56:
        import models.dcgan_56 as model
    elif opt.image_width == 64:
        import models.dcgan_64 as model
    elif opt.image_width == 128:
        import models.dcgan_128 as model
    else:
        raise ValueError(opt.image_width)
elif opt.model == 'vgg':
    if opt.image_width == 64:
        import models.vgg_64 as model
    elif opt.image_width == 128:
        import models.vgg_128 as model
    else:
        raise ValueError(opt.image_width)
else:
    raise ValueError('Unknown model: %s' % opt.model)

if opt.model_dir != '':
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']
else:
    encoder = model.encoder(opt.g_dim, opt.channels)
    decoder = model.decoder(opt.g_dim, opt.channels)
    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)
num_params_enc = utils.numel(encoder)
num_params_dec = utils.numel(decoder)
#print(f'\n{encoder}\n')
#print(f'{decoder}')
print('\nNumber of parameters:')
print(f'  encoder: {num_params_enc}')
print(f'  decoder: {num_params_dec}')

frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
posterior_optimizer = opt.optimizer(posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
prior_optimizer = opt.optimizer(prior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()
def kl_criterion(mu1, logvar1, mu2, logvar2):
    # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) = log( sqrt(
    # Daniel: unlike in FP case, here we have two non-identity Gaussians.
    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    return kld.sum() / opt.batch_size

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
prior.cuda()
encoder.cuda()
decoder.cuda()
mse_criterion.cuda()

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)

def get_training_batch():
    while True:
        for sequence in train_loader:
            if opt.action_cond:
                # Will return `batch` as a two-item tuple
                batch = utils.normalize_data(opt, dtype, sequence=sequence[0], sequence_acts=sequence[1])
            else:
                batch = utils.normalize_data(opt, dtype, sequence)
            yield batch
training_batch_generator = get_training_batch()

def get_testing_batch():
    while True:
        for sequence in test_loader:
            if opt.action_cond:
                # Will return `batch` as a two-item tuple
                batch = utils.normalize_data(opt, dtype, sequence=sequence[0], sequence_acts=sequence[1])
            else:
                batch = utils.normalize_data(opt, dtype, sequence)
            yield batch
testing_batch_generator = get_testing_batch()


# -------------------------------- plotting funtions ------------------------------------ #
def plot(x, epoch):
    """Forms the sample_{epoch}.{gif,png} files.

    Shows the results of evaluation time, using n_eval, which may differ from
    training, which can condition and predict on different numbers of images.
    Example, x_in = decoder(...) prediction has size: (batch, nchannel, H, W).
    """
    nsample = 20
    gen_seq = [[] for i in range(nsample)]
    gt_seq = [x[i] for i in range(len(x))]

    # Generate samples from model.
    for s in range(nsample):
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        gen_seq[s].append(x[0])
        x_in = x[0]
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if opt.last_frame_skip or i < opt.n_past:
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < opt.n_past:
                h_target = encoder(x[i])
                h_target = h_target[0].detach()
                z_t, _, _ = posterior(h_target)
                prior(h)
                frame_predictor(torch.cat([h, z_t], 1))
                x_in = x[i]
                gen_seq[s].append(x_in)
            else:
                z_t, _, _ = prior(h)
                h = frame_predictor(torch.cat([h, z_t], 1)).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq[s].append(x_in)

    # Plotting / GIF.
    to_plot = []
    gifs = [ [] for t in range(opt.n_eval) ]
    nrow = min(opt.batch_size, 10)

    for i in range(nrow):
        # ground truth sequence [Daniel: gt_seq[t][i].data for VSF data is shape (4,56,56)]
        row = []
        for t in range(opt.n_eval):
            row.append(gt_seq[t][i])
        to_plot.append(row)

        # best sequence
        min_mse = 1e12
        for s in range(nsample):
            mse = 0
            for t in range(opt.n_eval):
                mse +=  torch.sum( (gt_seq[t][i].data.cpu() - gen_seq[s][t][i].data.cpu())**2 )
            if mse < min_mse:
                min_mse = mse
                min_idx = s

        s_list = [min_idx,
                  np.random.randint(nsample),
                  np.random.randint(nsample),
                  np.random.randint(nsample),
                  np.random.randint(nsample)]
        for ss in range(len(s_list)):
            s = s_list[ss]
            row = []
            for t in range(opt.n_eval):
                row.append(gen_seq[s][t][i])
            to_plot.append(row)
        for t in range(opt.n_eval):
            row = []
            row.append(gt_seq[t][i])
            for ss in range(len(s_list)):
                s = s_list[ss]
                row.append(gen_seq[s][t][i])
            gifs[t].append(row)

    fname = '%s/gen/sample_%d.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)
    fname = '%s/gen/sample_%d.gif' % (opt.log_dir, epoch)
    utils.save_gif(fname, gifs)


def plot_rec(x, epoch):
    """Daniel: forms the rec_{epoch}.png files.

    Condition on first n_past frames, predict the next n_future frames. Example:
    for SM-MNIST, condition 5, predict the next 10. With a properly trained SVG-LP,
    rec_0.png has blurry (but promising) images, rec_299.png are very crisp. Makes
    figure with 1 row per item in the test minibatch. Put images in `gen_seq`:

    gen_seq = [ x[0], ..., x[n_past - 1], x[npast], ..., x[npast + nfuture - 1] ]
                                         ^ --------- decoded images ---------- ^

    As with train(), `i` means sampling z_i, PREDICTING x_i. Get E(x_i), decode to x_i.
    Here, `i < opt.n_past` is also changed to <= for the skip portion, but not later
    for gen_seq appending, since `i < opt.n_past` means we're on a ground truth image.
    Either way, we always call the frame predictor -- it's just that we can safely
    ignore its output if we're still in the ground truth generation phase, since the
    hidden state of the frame predictor is continually maintained internally! Great.

    BUT, this is actually using the ground truth images even after opt.n_past-1, as
    it's using encoder(x[i]) for all t here? I thought we were going to pass in the
    images in the `gen_seq` to the encoder? Well, that's for the plot() method above.
    Keep that in mind when reading these images!
    """
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    gen_seq = []
    gen_seq.append(x[0])
    x_in = x[0]

    for i in range(1, opt.n_past+opt.n_future):
        h = encoder(x[i-1])
        h_target = encoder(x[i])
        if opt.last_frame_skip or i <= opt.n_past:  # Daniel: see issue report on GitHub
            h, skip = h
        else:
            h, _ = h
        h_target, _ = h_target
        h = h.detach()
        h_target = h_target.detach()
        z_t, _, _= posterior(h_target)
        if i < opt.n_past:  # Daniel: should be correct
            frame_predictor(torch.cat([h, z_t], 1))
            gen_seq.append(x[i])
        else:
            h_pred = frame_predictor(torch.cat([h, z_t], 1))
            x_pred = decoder([h_pred, skip]).detach()
            gen_seq.append(x_pred)

    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(opt.n_past+opt.n_future):
            row.append(gen_seq[t][i])
        to_plot.append(row)
    fname = '%s/gen/rec_%d.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)


# ------------------------------ training funtions ------------------------------------ #
def train(x):
    """Daniel: iteration `i` means sampling z_i, PREDICTING x_i. Get E(x_i), decode to x_i.

    I like writing h = E[x_{t-1}] to remind me that it's an encoding.
    Shapes for frame predictor stuff:
        Assume g_dim (hidden variable) dim. is 128, and latent variable dim. is 10.
        h: (B, 128), z_t: (B, 10)
        FramePredictor input: (B, 138), output is back to (B, 128).

    There is one condition which helps to preserve `skip` layers, which means we can
    copy over stuff from given images (n_past) and use them later (n_future) because
    the for loop will NOT update `skip` any longer after we exceed the if condition case.

    BUT ... as mentioned: https://github.com/edenton/svg/issues/1 it seems like we want
    one more image's information, right? If n_past=2 that means x_0 and x_1 are ground truth
    frames, and we're predicting x_2 and beyond. But the for loop will stop updating `skip`
    when i=1, meaning that the last GROUND TRUTH is x_0 because given iteration `i`, we are
    actually predicting x_i (while conditioning on x_{i-1}). I agree!
    """
    frame_predictor.zero_grad()
    posterior.zero_grad()
    prior.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()

    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    prior.hidden = prior.init_hidden()

    mse = 0
    kld = 0
    for i in range(1, opt.n_past+opt.n_future):
        h = encoder(x[i-1])
        h_target = encoder(x[i])[0]
        if opt.last_frame_skip or i <= opt.n_past:
            h, skip = h
        else:
            h = h[0]
        z_t, mu, logvar = posterior(h_target)
        _, mu_p, logvar_p = prior(h)
        h_pred = frame_predictor(torch.cat([h, z_t], 1))
        x_pred = decoder([h_pred, skip])
        mse += mse_criterion(x_pred, x[i])
        kld += kl_criterion(mu, logvar, mu_p, logvar_p)

    loss = mse + kld*opt.beta
    loss.backward()

    frame_predictor_optimizer.step()
    posterior_optimizer.step()
    prior_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return mse.data.cpu().numpy()/(opt.n_past+opt.n_future), kld.data.cpu().numpy()/(opt.n_future+opt.n_past)


# ------------------------------- TRAINING LOOP ------------------------------------ #
LOSSES = defaultdict(list)

for epoch in range(opt.niter):
    frame_predictor.train()
    posterior.train()
    prior.train()
    encoder.train()
    decoder.train()
    epoch_mse = 0
    epoch_kld = 0
    progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()

    for i in range(opt.epoch_size):
        progress.update(i+1)
        # Daniel: x is list, len=seq_len, each item is of size (batch,channels,width,height).
        # SM-MNIST: seq_len=15, size of (for example) x[0] is (100,1,64,64). All torch.float32,
        # and all on cuda (check with `x[0].is_cuda`). If actions, this is a tuple.
        x = next(training_batch_generator)

        # train frame_predictor
        if opt.action_cond:
            x_imgs, x_acts = x
            mse, kld = train(x_imgs)  # TODO(daniel)
        else:
            mse, kld = train(x)
        epoch_mse += mse
        epoch_kld += kld

    progress.finish()
    utils.clear_progressbar()
    print('[%02d] mse loss: %.5f | kld loss: %.5f (%d)' % (epoch,
            epoch_mse/opt.epoch_size, epoch_kld/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))
    LOSSES['epoch'].append(epoch)
    LOSSES['mse_loss'].append(epoch_mse/opt.epoch_size)
    LOSSES['kld_loss'].append(epoch_kld/opt.epoch_size)
    LOSSES['tot_train'].append(epoch*opt.epoch_size*opt.batch_size)

    # plot some stuff [Daniel: I set encoder and decoder to use eval(), I can't see the harm?]
    frame_predictor.eval()
    encoder.eval()
    decoder.eval()
    posterior.eval()
    prior.eval()

    x = next(testing_batch_generator)
    if opt.action_cond:
        x_imgs, x_acts = x
        plot(x_imgs, epoch)
        plot_rec(x_imgs, epoch)
    else:
        plot(x, epoch)
        plot_rec(x, epoch)

    # save the model
    torch.save({
        'encoder': encoder,
        'decoder': decoder,
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'prior': prior,
        'opt': opt},
        '%s/model.pth' % opt.log_dir)
    if epoch % 10 == 0:
        print('log dir: %s' % opt.log_dir)

    # Save losses.
    loss_pth = '%s/losses.pkl' % opt.log_dir
    with open(loss_pth, 'wb') as fh:
        pickle.dump(LOSSES, fh)