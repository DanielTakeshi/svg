"""New architecture to handle 56x56 fabrics. But also test with SM-MNIST.

Parameter counts for 1 channel:
  encoder: 3805376
  decoder: 6558529

Recall, at least for PyTorch 1.0:
    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
"""
import torch
import torch.nn as nn

class dcgan_conv(nn.Module):

    def __init__(self, nin, nout, kernel_size=4, stride=2, padding=1):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):

    def __init__(self, nin, nout, kernel_size=4, stride=2, padding=1, output_padding=0):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, kernel_size=kernel_size, stride=stride,
                        padding=padding, output_padding=output_padding),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)


class encoder(nn.Module):
    """Daniel: to handle 56x56 images, we add 1 more padding to first 3 convs.

    Hacky, but the final dimensions are the same, here's the forward pass with 1 channel
    assuming SM-MNIST with 56x56 images:

    input:   torch.Size([100, 1, 56, 56])
    h1:      torch.Size([100, 64, 29, 29])
    h2:      torch.Size([100, 128, 15, 15])
    h3:      torch.Size([100, 256, 8, 8])
    h4:      torch.Size([100, 512, 4, 4])
    h5:      torch.Size([100, 128, 1, 1])
    h5.view: torch.Size([100, 128])

    And contrast that with the documentation for DCGAN 64.
    """

    def __init__(self, dim, nc=1):
        super(encoder, self).__init__()
        self.dim = dim
        nf = 64
        self.c1 = dcgan_conv(nc,     nf,     padding=2)
        self.c2 = dcgan_conv(nf,     nf * 2, padding=2)
        self.c3 = dcgan_conv(nf * 2, nf * 4, padding=2)
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim), [h1, h2, h3, h4]


class decoder(nn.Module):
    """Daniel: to handle 56x56 images, I had to first go through the inverse of what
    the `Conv2d`s did and have padding=2 to the last three layers. But then we also
    make use of PyTorch's special output_padding variable which can add a padding of 1
    to one side, which resolves ambiguity when dealing with stride > 1. See:

    https://pytorch.org/docs/1.0.0/nn.html?highlight=conv2d#torch.nn.Conv2d
    https://pytorch.org/docs/1.0.0/nn.html?highlight=conv2d#torch.nn.ConvTranspose2d

    So far it seems to at least run, here are shapes in the decoder for SM-MNIST:

    vec:    torch.Size([100, 128])
    d1:     torch.Size([100, 512, 4, 4])
    d2:     torch.Size([100, 256, 8, 8])
    d3:     torch.Size([100, 128, 15, 15])
    d4:     torch.Size([100, 64, 29, 29])
    output: torch.Size([100, 1, 56, 56])

    Contrast with the above. I was getting errors if the shapes did not align with
    the forward pass, since we have skip connections (a nice sanity check).
    """

    def __init__(self, dim, nc=1):
        super(decoder, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )
        self.upc2 = dcgan_upconv(nf * 8 * 2, nf * 4)
        self.upc3 = dcgan_upconv(nf * 4 * 2, nf * 2, padding=2, output_padding=1)
        self.upc4 = dcgan_upconv(nf * 2 * 2, nf,     padding=2, output_padding=1)
        self.upc5 = nn.Sequential(
                nn.ConvTranspose2d(nf * 2, nc, 4, 2, padding=2),
                nn.Sigmoid()
                )

    def forward(self, input):
        vec, skip = input
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1))
        d2 = self.upc2(torch.cat([d1, skip[3]], 1))
        d3 = self.upc3(torch.cat([d2, skip[2]], 1))
        d4 = self.upc4(torch.cat([d3, skip[1]], 1))
        output = self.upc5(torch.cat([d4, skip[0]], 1))
        return output
