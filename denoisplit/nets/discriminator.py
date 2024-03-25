"""
This part of the code is built based on the project:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import functools
from operator import imod

import torch
import torch.nn as nn


class Identity(nn.Module):
    def forward(self, x):
        return x


class Reshape(nn.Module):
    def forward(self, inp):
        return inp.view(inp.shape[0], -1)


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', input_hw=None, dense_ch_list=None, cnn_out_ch=None):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        input_hw           -- input spatial size. We assume a square image.
        dense_ch_list      -- list of dense channels
        cnn_out_ch         -- output channel of the CNN subunit.
    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(
            input_nc,
            ndf,
            n_layers=3,
            norm_layer=norm_layer,
            input_hw=input_hw,
            dense_ch_list=dense_ch_list,
            cnn_out_ch=cnn_out_ch,
        )
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(
            input_nc,
            ndf,
            n_layers_D,
            norm_layer=norm_layer,
            input_hw=input_hw,
            dense_ch_list=dense_ch_list,
            cnn_out_ch=cnn_out_ch,
        )
    elif netD == 'pixel':  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return net


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    def __init__(self,
                 input_nc,
                 ndf=64,
                 n_layers=3,
                 norm_layer=nn.BatchNorm2d,
                 input_hw=None,
                 dense_ch_list=None,
                 cnn_out_ch: int = None):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
            dense_ch_list   -- If we want to add a dense layer at the end, we provide here the list of channels for the dnese layer.
            cnn_out_ch      -- output channels for the CNN portion.
        """
        super(NLayerDiscriminator, self).__init__()
        self.input_hw = input_hw
        self.dense_ch_list = dense_ch_list

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 2
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, cnn_out_ch, kernel_size=kw, stride=1,
                               padding=padw)]  # output 1 channel prediction map
        self.cnn_model = nn.Sequential(*sequence)
        # dense portion now
        if self.dense_ch_list is not None:
            self.add_dense_layers(input_hw, input_nc, cnn_out_ch)
        else:
            self.model = self.cnn_model

    def add_dense_layers(self, input_hw, input_nc, cnn_out_ch):
        # finding the shape of the output coming out of CNN module
        with torch.no_grad():
            inp = torch.rand(1, input_nc, input_hw, input_hw)
            hw = self.cnn_model(inp).shape[-1]
        # the last channel is 1
        dense = self.get_dense(hw * hw * cnn_out_ch, self.dense_ch_list + [1])
        self.model = nn.Sequential(self.cnn_model, Reshape(), dense)

    def get_dense(self, in_channels, fc_list):
        modules = []
        for i, fc in enumerate(fc_list):
            modules.append(nn.Linear(in_channels, fc))
            if i < len(fc_list) - 1:
                modules.append(nn.LeakyReLU(0.2, True))
            in_channels = fc
        return nn.Sequential(*modules)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
