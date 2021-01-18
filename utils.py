import math
import torch
import socket
import argparse
import os
import cv2
import numpy as np
from sklearn.manifold import TSNE
import scipy.misc
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import functools
from skimage.measure import compare_psnr as psnr_metric
from skimage.measure import compare_ssim as ssim_metric
from scipy import signal
from scipy import ndimage
from PIL import Image, ImageDraw
from torchvision import datasets, transforms
from torch.autograd import Variable
import imageio

hostname = socket.gethostname()

def load_dataset(opt):
    """Machinery for train/test data. Need special cases for each data type.

    See the custom classes for fabric datasets, etc. The sequence length should be
    the combination of past + future (i.e., context + predicted output).
    """
    if opt.dataset == 'smmnist':
        from data.moving_mnist import MovingMNIST
        train_data = MovingMNIST(
                train=True,
                data_root=opt.data_root,
                seq_len=opt.n_past+opt.n_future,
                image_size=opt.image_width,
                deterministic=False,
                num_digits=opt.num_digits,
                channels=opt.channels)
        test_data = MovingMNIST(
                train=False,
                data_root=opt.data_root,
                seq_len=opt.n_eval,
                image_size=opt.image_width,
                deterministic=False,
                num_digits=opt.num_digits,
                channels=opt.channels)
    elif opt.dataset == 'bair':
        from data.bair import RobotPush
        train_data = RobotPush(
                train=True,
                data_root=opt.data_root,
                seq_len=opt.n_past+opt.n_future,
                image_size=opt.image_width)
        test_data = RobotPush(
                train=False,
                data_root=opt.data_root,
                seq_len=opt.n_eval,
                image_size=opt.image_width)
    elif opt.dataset == 'kth':
        from data.kth import KTH
        train_data = KTH(
                train=True,
                data_root=opt.data_root,
                seq_len=opt.n_past+opt.n_future,
                image_size=opt.image_width)
        test_data = KTH(
                train=False,
                data_root=opt.data_root,
                seq_len=opt.n_eval,
                image_size=opt.image_width)
    elif opt.dataset in ['fabric-random', 'fabric-01_2021']:
        # fabric-random: used in RSS 2020. Hard-coding image size of 56x56. We can
        # try use_actions=False to test, but we really need it True for real results.
        # Other version is fabric-01_2021. We can detect which one from the tail of
        # opt.data_root.
        from data.fabrics import FabricsData
        train_data = FabricsData(
                train=True,
                data_root=opt.data_root,
                seq_len=opt.n_past+opt.n_future,
                image_size=56,
                use_actions=opt.action_cond)
        test_data = FabricsData(
                train=False,
                data_root=opt.data_root,
                seq_len=opt.n_eval,
                image_size=56,
                use_actions=opt.action_cond)
    else:
        raise ValueError(f'{opt.dataset} not supported')

    return train_data, test_data

def sequence_input(seq, dtype):
    """Daniel: deprecated, but I don't want to mess with working code."""
    return [Variable(x.type(dtype)) for x in seq]

def normalize_data(opt, dtype, sequence, sequence_acts=None):
    """I don't think this is normalizing, just transposing?

    For video prediction, don't we just want to predict raw values? Though normalizing
    could be de-normalized if the process is deterministic. The bigger issue here is
    we need to get appropriate tensor shapes. For SM-MNIST, the transpose pattern is:
        (100, 15, 64, 64, 1) --> (15, 100, 64, 64, 1) --> (15, 100, 1, 64, 64)
    to get sequence length in the leading axis, followed by batch size, etc. Fabrics:
        (100, 10, 56, 56, 4) --> (10, 100, 56, 56, 4) --> (10, 100, 4, 56, 56)

    Args:
        dtype: torch.cuda.FloatTensor
        sequence: torch.Tensor (torch.float32) with the image data. Shape will be
            (batch_size, seq_len, height, width, channels). Example, with defaults
            for fabrics, (100,10,56,56,4) and with SM-MNIST, (100,15,64,64,1). They
            have the same set of shapes so we should follow the same convention.
        sequence_acts: torch.Tensor (torch.float32) with action data. Shape will be
            (batch_size, seq_len-1, channels), so swap the first two of these to get
            (seq_len-1, batch_size, channels). See documentation in fabrics.py file
            for the rationale. WE THEN IGNORE THE FIRST `opt.n_past-1` actions, so
            that the FIRST action in sequence_acts will be aligned with the current
            input observation (the LAST of the `opt.n_past` context frames). UPDATE:
            no this is bad, we actually do need the first few actions so that we can
            get consistent input dimensions during ground truth stage. AH!
    """
    if opt.dataset in ['smmnist', 'kth', 'bair', 'fabric-random', 'fabric-01_2021']:
        sequence.transpose_(0, 1)
        sequence.transpose_(3, 4).transpose_(2, 3)
    else:
        sequence.transpose_(0, 1)

    if sequence_acts is not None:
        assert opt.dataset in ['fabric-random', 'fabric-01_2021'], opt.dataset
        sequence_imgs = sequence_input(sequence, dtype) # same as usual
        sequence_acts.transpose_(0, 1)                  # new for actions
        #sequence_acts = sequence_acts[opt.n_past-1 : ]  # DO NOT DO THIS
        return (sequence_imgs, sequence_acts)
    else:
        return sequence_input(sequence, dtype)

def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))

def image_tensor(inputs, padding=1):
    """Daniel: returns torch tensor of (c_dim, ...), so we should remove depth
    channel(s) when making visualizations."""
    # assert is_sequence(inputs)
    assert len(inputs) > 0

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        # Daniel: I'm only seeing images[0].dim() == 3 for fabrics-random.
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # Daniel: I'm only seeing images[0].dim() == 3 for fabrics-random.
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                   (i+1) * y_dim + i * padding].copy_(image)
        return result

def save_np_img(fname, x):
    if x.shape[0] == 1:
        x = np.tile(x, (3, 1, 1))
    img = scipy.misc.toimage(x,
                             high=255*x.max(),
                             channel_axis=0)
    img.save(fname)

def make_image(tensor):
    """Daniel: if channels > 3, get rid of anything after first 3.

    BTW: this function is deprecated, we should be using Image.fromarray(...):
    https://docs.scipy.org/doc/scipy-1.2.1/reference/generated/scipy.misc.toimage.html
    I _believe_ scipy.misc.toimage returns in RGB mode. I'm saving using OpenCV,
    and that will save these images in BGR mode.
    """
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    elif tensor.size(0) > 3:
        tensor = tensor[:3, :, :]
    tensor = tensor.detach().numpy()
    return scipy.misc.toimage(tensor,
                              high=255*tensor.max(),
                              channel_axis=0)

def draw_text_tensor(tensor, text):
    np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
    pil = Image.fromarray(np.uint8(np_x*255))
    draw = ImageDraw.Draw(pil)
    draw.text((4, 64), text, (0,0,0))
    img = np.asarray(pil)
    return Variable(torch.Tensor(img / 255.)).transpose(1, 2).transpose(0, 1)

def save_gif(filename, inputs, duration=0.25):
    """Daniel: saves a GIF, called from `plot()` function.

    I was getting a warning:
        Lossy conversion from float32 to uint8. Range [0, 1].
        Convert image to uint8 prior to saving to suppress this warning.

    To fix that, I'm making the numpy images uint8. Since the original images are in
    range [0,1] then scale by 255 before turning to np.uint8. They get clamped here,
    see the `clamp(0,1)` call.

    After `image_tensor()`, remove the depth channels for fabrics data. We'll assume
    we can do this by only keeping the leading 3 dimensions. Here, after img.cpu(), shapes:

        [1, 640, 389] for 64x64 SM-MNIST
        [1, 560, 341] for 56x56 SM-MNIST
        [4, 560, 341] for 56x56 fabric-random

    Update: actually the `np_image` that gets created after numpy-ing it will be put in
    imageio.mimsave, and that saves in RGB mode. So we want to save in BGR mode to be
    consistent with how we save with cv2 everywhere else, so we can convert.
    """
    def keep_only_rgb(x):
        assert x.dim() == 3, x.dim()
        x = x[:3, :, :]
        return x

    images = []
    for tensor in inputs:
        img = image_tensor(tensor, padding=0)
        img = img.cpu()
        if img.size(0) > 3:
            img = keep_only_rgb(img)
        img = img.transpose(0,1).transpose(1,2).clamp(0,1)
        np_image = (img.numpy() * 255.0).astype(np.uint8)

        if np_image.shape[2] == 3:
            #np_image = np_image[:,:,::-1]  # Daniel: also works
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        images.append(np_image)
    imageio.mimsave(filename, images, duration=duration)

def save_gif_with_text(filename, inputs, text, duration=0.25):
    images = []
    for tensor, text in zip(inputs, text):
        img = image_tensor([draw_text_tensor(ti, texti) for ti, texti in zip(tensor, text)], padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1).numpy()
        images.append(img)
    imageio.mimsave(filename, images, duration=duration)

def save_image(filename, tensor):
    """Save an image to `filename`.

    Daniel: in our VSF code, we're saving using cv2.imwrite(), which saves in BGR.
    But img.save uses PIL and that will save in RGB mode, which makes it hard to compare.
    Therefore we should use cv2 whenever we can.

    See refs such as:
    https://note.nkmk.me/en/python-opencv-bgr-rgb-cvtcolor/
    https://stackoverflow.com/questions/4661557/pil-rotate-image-colors-bgr-rgb
    """
    img = make_image(tensor)
    #img.save(filename)  # Daniel: I replaced this with the next two lines.
    numpy_image = np.asarray(img)
    cv2.imwrite(filename, numpy_image)

def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(filename, images)

def prod(l):
    return functools.reduce(lambda x, y: x * y, l)

def batch_flatten(x):
    return x.resize(x.size(0), prod(x.size()[1:]))

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            for c in range(gt[t][i].shape[0]):
                ssim[i, t] += ssim_metric(gt[t][i][c], pred[t][i][c])
                psnr[i, t] += psnr_metric(gt[t][i][c], pred[t][i][c])
            ssim[i, t] /= gt[t][i].shape[0]
            psnr[i, t] /= gt[t][i].shape[0]
            mse[i, t] = mse_metric(gt[t][i], pred[t][i])

    return mse, ssim, psnr

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            for c in range(gt[t][i].shape[0]):
                res = finn_ssim(gt[t][i][c], pred[t][i][c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(gt[t][i][c], pred[t][i][c])
            ssim[i, t] /= gt[t][i].shape[0]
            psnr[i, t] /= gt[t][i].shape[0]
            mse[i, t] = mse_metric(gt[t][i], pred[t][i])

    return mse, ssim, psnr

def finn_psnr(x, y):
    mse = ((x - y)**2).mean()
    return 10*np.log(1/mse)/np.log(10)

def gaussian2(size, sigma):
    A = 1/(2.0*np.pi*sigma**2)
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = A*np.exp(-((x**2/(2.0*sigma**2))+(y**2/(2.0*sigma**2))))
    return g

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def finn_ssim(img1, img2, cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 1 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`

    https://stackoverflow.com/questions/49201236/
    """
    parameters = m.parameters()
    if only_trainable:
        parameters = list(p for p in parameters if p.requires_grad)
    unique = dict((p.data_ptr(), p) for p in parameters).values()
    return sum(p.numel() for p in unique)
