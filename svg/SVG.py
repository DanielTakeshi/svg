"""Put things in a class to make it easier to save/load in other projects."""
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
from svg import utils
import svg.models.lstm as lstm_models
import itertools
import progressbar
import numpy as np
import sys
import pickle
from collections import defaultdict


class SVG:
    """Follows https://pytorch.org/tutorials/beginner/saving_loading_models.html

    The __init__ reconstructs models from classes, and if desired (if model_dir != ''),
    we use the load_state_dict method. Use the same `model_dir != ''` condition to check
    if we should do other stuff, such as avoid making the train/test data loaders.
    """

    def __init__(self, opt):
        self.opt = opt

        if opt.model_dir != '':
            # load model ONLY FOR INFERENCE, not for continued training
            checkpoint = torch.load(opt.model_dir) # pass in ENTIRE path, including specific epoch if desired.
            optimizer = opt.optimizer
            model_dir = opt.model_dir
            opt = checkpoint['opt']     # Pretty sure we need this to properly define our classes before loading state dicts
            self.opt = opt              # Actually this won't matter too much, but I think it's safe to have.
            opt.optimizer = optimizer
            opt.model_dir = model_dir
            opt.batch_size = 1          # NOTE Let's override this FOR INFERENCE since otherwise the RNNs expect the same train batch size.
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
        self.dtype = torch.cuda.FloatTensor
        print(opt)

        # -------------------------------------- optimizers ------------------------------------- #
        if opt.optimizer == 'adam':
            opt.optimizer = optim.Adam
        elif opt.optimizer == 'rmsprop':
            opt.optimizer = optim.RMSprop
        elif opt.optimizer == 'sgd':
            opt.optimizer = optim.SGD
        else:
            raise ValueError('Unknown optimizer: %s' % opt.optimizer)

        # --------------------------------------- MODELS --------------------------------------------- #
        # Ok now here we can't just do `saved_model['prior']` or `checkpoint['prior']`, we have to
        # build the classes for the models first, then we can load their state dicts. Fortunately
        # the `opt` gives us all we need.
        in_frame = opt.g_dim + opt.z_dim
        in_postr = opt.g_dim
        in_prior = opt.g_dim
        if opt.action_cond:
            in_frame += opt.act_dim
            #in_postr += opt.act_dim # NOT the posterior!
            in_prior += opt.act_dim
        self.frame_predictor = lstm_models.lstm(         in_frame, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
        self.posterior       = lstm_models.gaussian_lstm(in_postr, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
        self.prior           = lstm_models.gaussian_lstm(in_prior, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers,     opt.batch_size)

        # HERE is where we can load.
        if opt.model_dir != '':
            self.frame_predictor.load_state_dict( checkpoint['frame_predictor'] )
            self.posterior.load_state_dict( checkpoint['posterior'] )
            self.prior.load_state_dict( checkpoint['prior'] )
        else:
            self.frame_predictor.apply(utils.init_weights)
            self.posterior.apply(utils.init_weights)
            self.prior.apply(utils.init_weights)

        if opt.model == 'dcgan':
            if opt.image_width == 56:
                import svg.models.dcgan_56 as model
            elif opt.image_width == 64:
                import svg.models.dcgan_64 as model
            elif opt.image_width == 128:
                import svg.models.dcgan_128 as model
            else:
                raise ValueError(opt.image_width)
        elif opt.model == 'vgg':
            if opt.image_width == 64:
                import svg.models.vgg_64 as model
            elif opt.image_width == 128:
                import svg.models.vgg_128 as model
            else:
                raise ValueError(opt.image_width)
        else:
            raise ValueError('Unknown model: %s' % opt.model)

        # Now the last two model components.
        self.encoder = model.encoder(opt.g_dim, opt.channels)
        self.decoder = model.decoder(opt.g_dim, opt.channels)
        if opt.model_dir != '':
            self.decoder.load_state_dict( checkpoint['decoder'] )
            self.encoder.load_state_dict( checkpoint['encoder'] )
        else:
            self.encoder.apply(utils.init_weights)
            self.decoder.apply(utils.init_weights)

        num_params_enc = utils.numel(self.encoder)
        num_params_dec = utils.numel(self.decoder)
        print('\nNumber of parameters:')
        print(f'  encoder: {num_params_enc}')
        print(f'  decoder: {num_params_dec}')

        # Optimizers for the models.
        self.frame_predictor_optimizer = opt.optimizer(self.frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.posterior_optimizer       = opt.optimizer(self.posterior.parameters(),       lr=opt.lr, betas=(opt.beta1, 0.999))
        self.prior_optimizer           = opt.optimizer(self.prior.parameters(),           lr=opt.lr, betas=(opt.beta1, 0.999))
        self.encoder_optimizer         = opt.optimizer(self.encoder.parameters(),         lr=opt.lr, betas=(opt.beta1, 0.999))
        self.decoder_optimizer         = opt.optimizer(self.decoder.parameters(),         lr=opt.lr, betas=(opt.beta1, 0.999))

        # Transfer to GPU.
        self.frame_predictor.cuda()
        self.posterior.cuda()
        self.prior.cuda()
        self.encoder.cuda()
        self.decoder.cuda()

        # Define MSE criterion, also transfer to GPU.
        self.mse_criterion = nn.MSELoss()
        self.mse_criterion.cuda()

        # For data loading.
        if opt.model_dir != '':
            print('opt.model_dir is not empty, hence avoid making train/test data loaders')
        else:
            train_data, test_data = utils.load_dataset(opt)
            self.train_loader = DataLoader(train_data,
                                           num_workers=opt.data_threads,
                                           batch_size=opt.batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           pin_memory=True)
            self.test_loader = DataLoader(test_data,
                                          num_workers=opt.data_threads,
                                          batch_size=opt.batch_size,
                                          shuffle=True,
                                          drop_last=True,
                                          pin_memory=True)

    def kl_criterion(self, mu1, logvar1, mu2, logvar2):
        # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) = log( sqrt(
        # Daniel: unlike in FP case, here we have two non-identity Gaussians.
        opt = self.opt
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
        return kld.sum() / opt.batch_size

    def get_mse_components(self, x, y, split=False):
        """Daniel: adding to compute MSEs (but avoid PyTorch optimization).
        https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
        Computes portion of the MSE contributed from RGB vs D components, sums to full MSE.
        Well, if split=True. If split=False then it's pretty easy.
        """
        assert x.is_cuda and y.is_cuda
        a = x.detach().clone().cpu().numpy()
        b = y.detach().clone().cpu().numpy()
        assert a.shape == b.shape
        if split:
            num_items = np.prod(a.shape)
            #mse_0 = ((a[:,:3,:,:] - b[:,:3,:,:]) ** 2).mean()  # No, will divide by a smaller number.
            #mse_1 = ((a[:,3:,:,:] - b[:,3:,:,:]) ** 2).mean()  # No, will divide by a smaller number.
            mse_0 = np.sum((a[:,:3,:,:] - b[:,:3,:,:]) ** 2) / num_items  # RGB 'portion'
            mse_1 = np.sum((a[:,3:,:,:] - b[:,3:,:,:]) ** 2) / num_items  # D 'portion'
            return (mse_0, mse_1)
        else:
            return ((a-b)**2).mean()

    def get_training_batch(self):
        opt = self.opt
        while True:
            for sequence in self.train_loader:
                if opt.action_cond:
                    # Will return `batch` as a two-item tuple
                    batch = utils.normalize_data(opt, self.dtype, sequence=sequence[0], sequence_acts=sequence[1])
                else:
                    batch = utils.normalize_data(opt, self.dtype, sequence)
                yield batch

    def get_testing_batch(self):
        opt = self.opt
        while True:
            for sequence in self.test_loader:
                if opt.action_cond:
                    # Will return `batch` as a two-item tuple
                    batch = utils.normalize_data(opt, self.dtype, sequence=sequence[0], sequence_acts=sequence[1])
                else:
                    batch = utils.normalize_data(opt, self.dtype, sequence)
                yield batch

    def plot(self, x, epoch, x_acts=None):
        """Forms the sample_{epoch}.{gif,png} files.

        Shows the results of evaluation time, using n_eval, which may differ from training,
        which can condition and predict on different numbers of images. For example,
        x_in = decoder(...) prediction has size: (batch, nchannel, H, W).

        Get `nsample` samples, meaning this many different instances of prediction, and for
        each of these, we predict frames in the future _starting_ from the ground truths.
        Unlike plot_rec(), this time after the ground truth images, SVG does NOT get further
        ground truth images and instead passes the predicted image to the encoder for the next
        time frame. Good! We also see this reflected in that `h_target = encoder(x[i])` is ONLY
        called if `i < opt.n_past`.

        NOTE: `opt.n_eval` is maybe misleading, I thought it meant predicting this many images.
        However, we actually use the first `opt.n_past` frames and treat those as ground truth,
        which is the same as `plot_rec`, then any REMAINING frames, `n_eval-n_past`, will be
        predicted. So for my fabric example with past=3, future=7, I set eval=10 so that it
        replicates the training setup, given the first 3 frames, predict the next 7.
        """
        opt = self.opt
        nsample = 20
        gen_seq = [[] for i in range(nsample)]
        gt_seq = [x[i] for i in range(len(x))]

        # Generate samples from model.
        for s in range(nsample):
            self.frame_predictor.hidden = self.frame_predictor.init_hidden()
            self.posterior.hidden       = self.posterior.init_hidden()
            self.prior.hidden           = self.prior.init_hidden()
            gen_seq[s].append(x[0])
            x_in = x[0]

            # Daniel: as usual, iter `i` means x_{i-1} is most recently conditioned frame, SVG
            # _samples_ z_i and x_i. If it's before n_past, just use ground truth x_i for image.
            for i in range(1, opt.n_eval):
                h = self.encoder(x_in)
                if opt.last_frame_skip or i <= opt.n_past:  # Daniel: see issue report on GitHub
                    h, skip = h
                else:
                    h, _ = h

                # Daniel: action conditioning. Updating h: (B,g_dim) --> (B, g_dim+4).
                # Remember, while we generate images, actions should be interpreted as given.
                # Given a_{i-1} we use that with Enc(prior_image) to get (predicted) x_i.
                if x_acts is not None:
                    x_a = x_acts[i-1].cuda()
                    h = torch.cat([h, x_a], dim=1)  # overrides h

                h = h.detach()
                if i < opt.n_past:
                    h_target = self.encoder(x[i])
                    h_target = h_target[0].detach()
                    z_t, _, _ = self.posterior(h_target)
                    self.prior(h)
                    self.frame_predictor(torch.cat([h, z_t], 1))
                    x_in = x[i]  # Daniel: GROUND TRUTH.
                    gen_seq[s].append(x_in)
                else:
                    # Daniel: the posterior isn't used, as it relies on ground truth images
                    # _after_ n_past. So instead we use the prior, which we cleverly kept
                    # updating during the ground truth stage (i < opt.n_past).
                    z_t, _, _ = self.prior(h)
                    h = self.frame_predictor(torch.cat([h, z_t], 1)).detach()  # overrides h (fine)
                    x_in = self.decoder([h, skip]).detach()
                    gen_seq[s].append(x_in)

        # Daniel: Plotting / GIF. For the .png, show (a) ground truth sequence, (b) the
        # predictions with minimum MSE, and (c) 4 more randomly drawn samples from SVG.
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

            # Daniel: this given code should be modified to make samples guaranteed as distinct.
            s_indices = np.random.permutation(5)  # Hard-coding 5 here. Change if desired.
            s_list = [min_idx]
            for s_idx in s_indices:
                if s_idx in s_list:
                    continue  # avoid equal to min_idx case
                if len(s_list) == 5:
                    continue
                s_list.append(s_idx)

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

    def plot_rec(self, x, epoch, x_acts=None):
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
        Keep that in mind when reading these images! Also, remember that the first few
        frames in the image, up to `n_past`, will be ground-truth. The `rec.png` image
        should have the first `n_past` frames look sharp since they are from the data.

        NOTE: the prior is not used at all here (why?), whereas plot uses the prior.

        Update Jan2021: let's change this so we return the MSEs of all the frames that
        are NOT ground truth, so compare decoded frame vs ground truth and average over
        the future frames (not past+future as in the train method).
        """
        opt = self.opt
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        self.posterior.hidden       = self.posterior.init_hidden()
        gen_seq = []
        gen_seq.append(x[0])
        x_in = x[0]
        mse_decode = 0

        for i in range(1, opt.n_past+opt.n_future):
            h = self.encoder(x[i-1])
            h_target = self.encoder(x[i])
            if opt.last_frame_skip or i <= opt.n_past:  # Daniel: see issue report on GitHub
                h, skip = h
            else:
                h, _ = h

            # Daniel: handle action conditioning. The prior isn't used at all, so we can
            # move this later (but ordering must be h, then x_a, then z_t).
            if x_acts is not None:
                x_a = x_acts[i-1].cuda()
                h = torch.cat([h, x_a], dim=1)

            h_target, _ = h_target
            h = h.detach()
            h_target = h_target.detach()
            z_t, _, _= self.posterior(h_target)
            if i < opt.n_past:  # Daniel: should be correct
                self.frame_predictor(torch.cat([h, z_t], 1))
                gen_seq.append(x[i])
            else:
                h_pred = self.frame_predictor(torch.cat([h, z_t], 1))
                x_pred = self.decoder([h_pred, skip]).detach()
                gen_seq.append(x_pred)

                # Daniel: handle decoder MSE predictions, happens n_future times.
                mse_decode += self.get_mse_components(x_pred, x[i], split=False)

        # Plotting.
        to_plot = []
        nrow = min(opt.batch_size, 10)
        for i in range(nrow):
            row = []
            for t in range(opt.n_past+opt.n_future):
                row.append(gen_seq[t][i])
            to_plot.append(row)
        fname = '%s/gen/rec_%d.png' % (opt.log_dir, epoch)
        utils.save_tensors_image(fname, to_plot)

        # New, returning 'validation' decoding. This will be slightly noisier than training
        # MSE since it's only one minibatch, whereas train MSE covers an entire epoch.
        return mse_decode / opt.n_future

    def train(self, x, x_acts=None):
        """Daniel: iteration `i` means sampling z_i, PREDICTING x_i. Get E(x_i), decode to x_i.

        Returns: a tuple
            (mse_loss, kld_loss) or (mse_loss, kld_loss, mse_rgb, mse_depth)

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

        Update Jan2021: now supports action-conditioning. See `data/fabrics.py` for details on
        data batches. Affects encoding for x_{i-1}, so we want action a_{i-1}. BTW we can always
        assume the 'ground truth' (it's not really ground truth) action is there, it's not what
        we're predicting and at test time we'd supply candidate sequences of actions. Also, for
        tracking RGB and D statistics, returns an extra item.
        """
        opt = self.opt
        self.frame_predictor.zero_grad()
        self.posterior.zero_grad()
        self.prior.zero_grad()
        self.encoder.zero_grad()
        self.decoder.zero_grad()

        # initialize the hidden state.
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        self.posterior.hidden       = self.posterior.init_hidden()
        self.prior.hidden           = self.prior.init_hidden()
        mse_c = 0
        mse_d = 0
        mse = 0
        kld = 0
        for i in range(1, opt.n_past+opt.n_future):
            h = self.encoder(x[i-1])
            h_target = self.encoder(x[i])[0]
            if opt.last_frame_skip or i <= opt.n_past:
                h, skip = h
            else:
                h = h[0]

            # Daniel: action conditioning. Updating h: (B,g_dim) --> (B, g_dim+4).
            if x_acts is not None:
                x_a = x_acts[i-1].cuda()
                h = torch.cat([h, x_a], dim=1)

            z_t, mu, logvar = self.posterior(h_target)
            _, mu_p, logvar_p = self.prior(h)
            h_pred = self.frame_predictor(torch.cat([h, z_t], 1))
            x_pred = self.decoder([h_pred, skip])
            mse += self.mse_criterion(x_pred, x[i])
            kld += self.kl_criterion(mu, logvar, mu_p, logvar_p)

            # Daniel: adding new statistics, assumes 4 channel images.
            if x_acts is not None:
                this_c, this_d = self.get_mse_components(x_pred, x[i], split=True)
                mse_c += this_c
                mse_d += this_d

        # Compute loss function and step optimizers.
        loss = mse + kld*opt.beta
        loss.backward()
        self.frame_predictor_optimizer.step()
        self.posterior_optimizer.step()
        self.prior_optimizer.step()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        # Daniel: shouldn't these be divided by npast + nfuture MINUS one? For loop starts at 1.
        mse_cpu = mse.data.cpu().numpy() / (opt.n_past + opt.n_future - 1)
        kld_cpu = kld.data.cpu().numpy() / (opt.n_past + opt.n_future - 1)
        if x_acts is not None:
            mse_c = mse_c / (opt.n_past + opt.n_future - 1)
            mse_d = mse_d / (opt.n_past + opt.n_future - 1)
            return (mse_cpu, kld_cpu, mse_c, mse_d)
        else:
            return (mse_cpu, kld_cpu)

    def train_loop(self):
        opt = self.opt
        LOSSES = defaultdict(list)
        training_batch_generator = self.get_training_batch()
        testing_batch_generator = self.get_testing_batch()

        for epoch in range(opt.niter):
            self.frame_predictor.train()
            self.posterior.train()
            self.prior.train()
            self.encoder.train()
            self.decoder.train()
            epoch_mse = 0
            epoch_kld = 0
            epoch_mse_c = 0
            epoch_mse_d = 0
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
                    mse, kld, mse_c, mse_d = self.train(x_imgs, x_acts=x_acts)
                    epoch_mse_c += mse_c
                    epoch_mse_d += mse_d
                else:
                    mse, kld = self.train(x)
                epoch_mse += mse
                epoch_kld += kld

            progress.finish()
            utils.clear_progressbar()
            if opt.action_cond:
                print('[%02d] mse loss: %.5f | C: %.5f, D: %.5f | kld loss: %.5f (%d)' % (epoch,
                        epoch_mse   / opt.epoch_size,
                        epoch_mse_c / opt.epoch_size,
                        epoch_mse_d / opt.epoch_size,
                        epoch_kld   / opt.epoch_size,
                        epoch * opt.epoch_size * opt.batch_size))
            else:
                print('[%02d] mse loss: %.5f | kld loss: %.5f (%d)' % (epoch,
                        epoch_mse / opt.epoch_size,
                        epoch_kld / opt.epoch_size,
                        epoch * opt.epoch_size * opt.batch_size))
            LOSSES['epoch'].append(epoch)
            LOSSES['mse_loss'].append(epoch_mse / opt.epoch_size)
            LOSSES['kld_loss'].append(epoch_kld / opt.epoch_size)
            LOSSES['tot_train'].append(epoch * opt.epoch_size * opt.batch_size)
            if opt.action_cond:
                LOSSES['mse_loss_C'].append(epoch_mse_c / opt.epoch_size)
                LOSSES['mse_loss_D'].append(epoch_mse_d / opt.epoch_size)

            # plot some stuff [Daniel: I set encoder and decoder to use eval(), I can't see the harm?]
            self.frame_predictor.eval()
            self.encoder.eval()
            self.decoder.eval()
            self.posterior.eval()
            self.prior.eval()

            # Daniel: should also be adding testing set statistics.
            x = next(testing_batch_generator)
            if opt.action_cond:
                x_imgs, x_acts = x
                self.plot(x_imgs, epoch, x_acts=x_acts)
                valid_mse = self.plot_rec(x_imgs, epoch, x_acts=x_acts)
            else:
                self.plot(x, epoch)
                valid_mse = self.plot_rec(x, epoch)
            LOSSES['valid_mse_loss_rec'].append(valid_mse)  # No need to divide by epoch
            print('[%02d] mse loss: %.5f [valid, rec]' % (epoch, valid_mse))

            # save the model
            if (epoch == opt.niter-1) or (epoch % 50 == 0):
                self.save(epoch)
            if epoch % 10 == 0:
                print('log dir: %s' % opt.log_dir)

            # Save losses.
            loss_pth = '%s/losses.pkl' % opt.log_dir
            with open(loss_pth, 'wb') as fh:
                pickle.dump(LOSSES, fh)

    def predict(self, x, x_acts, n_past=1, n_future=5):
        """Daniel: see predict_svg_lp.py, this should be dragged/dropped into cloth-visual-mpc."""
        opt = self.opt
        #assert opt.n_past == 1  # If predicting we want n_past=1 but usually training was different.
        # Actually that's a good point we should not be using the same opt, let's specify as args.
        gen_seq = []

        self.encoder.eval()
        self.decoder.eval()
        self.frame_predictor.eval()
        self.posterior.eval()
        self.prior.eval()
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        self.posterior.hidden       = self.posterior.init_hidden()
        self.prior.hidden           = self.prior.init_hidden()

        # Reshape single context frame. If we actually do more than 1 context frame, we should
        # instead format `x` like we normally do during training by having it as a list, etc.
        x_in = x.transpose(2,0,1)[np.newaxis,:]
        x_in = np.array(x_in).astype(np.float32) / 255.0     # Only major processing was div by 255.
        x_in = torch.from_numpy(x_in).cuda()

        # Remmeber, we normally have n_eval here, which is n_past + n_future.
        for i in range(1, n_past + n_future):
            h = self.encoder(x_in)
            if opt.last_frame_skip or i <= n_past:
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
            if i < n_past:
                h_target = self.encoder(x[i])
                h_target = h_target[0].detach()
                z_t, _, _ = self.posterior(h_target)
                self.prior(h)
                self.frame_predictor(torch.cat([h, z_t], 1))
                x_in = x[i]
            else:
                z_t, _, _ = self.prior(h)
                h = self.frame_predictor(torch.cat([h, z_t], 1)).detach()
                x_in = self.decoder([h, skip]).detach()
                gen_seq.append(x_in.data.cpu().numpy())

        # Reshape it to match desired output for consistency with SV2P.
        prediction = np.array(gen_seq)              # (n_future, 1, 4, 56, 56)
        prediction = np.squeeze(prediction)         # (n_future, 4, 56, 56)
        prediction = prediction.transpose(0,2,3,1)  # (n_future, 56, 56, 4)
        # NOTE this value is within (0,1), should multiply by 255 if desired.
        return prediction

    def save(self, epoch):
        """Save the CORRECT way using state dicts!
        We don't need to save optimizers; we're not resuming training, just doing inference.
        The `opt` will be very helpful, I believe we can load it and then it overrides stuff
        from loading, in case we make errors there?
        """
        opt = self.opt
        tail = 'model_{}.pth'.format( str(epoch).zfill(4) )
        model_path = os.path.join(opt.log_dir, tail)
        #torch.save({
        #    'encoder':          self.encoder,
        #    'decoder':          self.decoder,
        #    'frame_predictor':  self.frame_predictor,
        #    'posterior':        self.posterior,
        #    'prior':            self.prior,
        #    'opt':              opt},
        #    model_path)
        torch.save({
            'encoder':          self.encoder.state_dict(),
            'decoder':          self.decoder.state_dict(),
            'frame_predictor':  self.frame_predictor.state_dict(),
            'posterior':        self.posterior.state_dict(),
            'prior':            self.prior.state_dict(),
            'opt':              opt,
            }, model_path)


if __name__ == "__main__":
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
    parser.add_argument('--act_dim', type=int, default=4, help='Need to adjust architecture a bit')
    parser.add_argument('--act_design', type=int, default=1, help='TODO(daniel) if we try multiple designs.')
    opt = parser.parse_args()

    svg = SVG(opt)
    svg.train_loop()
