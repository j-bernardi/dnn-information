"""
Implementation of 3d U-Net
Based on: https://github.com/shiba24/3d-unet
From paper: https://arxiv.org/pdf/1606.06650.pdf

Note, this may also be useful:
https://github.com/jeffkinnison/unet/blob/master/pytorch/unet3d.py
"""

"""
TODO ? 
We introduced one extra residual connection within each block of layers, 
so that the output of each block consists of the sum of the features of the last layer, 
and the first layer of the block in which the features dimensions match. 
"""

import torch, gc
import torch.nn as nn
import torch.nn.functional as F
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UNet3D(nn.Module):
    """The network."""

    def __init__(self, in_channel=1, n_classes=15, device="find"):
        """
        Archtecture modelled from (refd in DeepMind Paper):
            https://arxiv.org/pdf/1606.06650.pdf
        Open source modelling:
            Based on: https://github.com/shiba24/3d-unet
        Default:
            Input: (default) 32-channel 448x512x9 voxels image
            Output: estimated probability over the 15 (default) classes
                (for each of the 448x512x1 output voxels)
        """

        # Down-sizes from the paper
        #self.sizes = [(0,0,0), (9, 448, 512), (9, 224, 256), (9, 112, 128), (9, 56, 64), 
        #              (7, 28, 32), (5, 14, 16), (3, 7, 8), (1, 1, 1)]
        #self.sizes2 = [(0,0,0), (1, 448, 512), (1, 224, 256), (1, 112, 128), (1, 56, 64), 
        #              (1, 28, 32), (1, 14, 16), (1, 7, 8), (1, 1, 1)]

        # Smaller down-sizes:
        self.sizes = [(0,0,0), (9, 112, 128), (9, 112, 128), (9, 112, 128), (9, 56, 64), 
                      (7, 28, 32), (5, 14, 16), (3, 7, 8), (1, 1, 1)]
        # Back up:
        self.sizes2 = [(0,0,0), (1, 112, 128), (1, 112, 128), (1, 112, 128), (1, 56, 64), 
                      (1, 28, 32), (1, 14, 16), (1, 7, 8), (1, 1, 1)]

        self.in_channel = in_channel
        self.n_classes = n_classes

        if device == "find":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else: 
            self.device = torch.device(device)

        super(UNet3D, self).__init__()

        ## ENCODING ##
        xy_kernel, z_kernel = (1,3,3), (3,1,1) 
        xy_padding = (0,1,1)
        bottom_channels = 256 # TEMP : 4096

        # Level 1
        # QUESTION: this to get 32 channels in layer 1?
        self.ec_init = self.encoder(self.in_channel, 32, padding=0, kernel_size=1, n_convs=1)
        self.ec11 = self.encoder(32, 32, padding=xy_padding, kernel_size=xy_kernel, n_convs=3)
        self.down12 = self.bilinear(32, 32, size=self.sizes[2])

        # Level 2
        self.ec22 = self.encoder(32, 32, padding=xy_padding, kernel_size=xy_kernel, n_convs=3)
        self.down23 = self.bilinear(32, 32, size=self.sizes[3])

        # Level 3
        self.ec33 = self.encoder(32, 32, padding=xy_padding, kernel_size=xy_kernel, n_convs=3)
        self.down34 = self.bilinear(32, 64, size=self.sizes[4])

        # Level 4
        self.ec441 = self.encoder(64, 64, padding=xy_padding, kernel_size=xy_kernel, n_convs=3) # green
        self.ec442 = self.encoder(64, 64, padding=0, kernel_size=z_kernel) # turquoise
        self.down45 = self.bilinear(64, 128, size=self.sizes[5])

        # Level 5
        self.ec551 = self.encoder(128, 128, padding=xy_padding, kernel_size=xy_kernel, n_convs=3) # green
        self.ec552 = self.encoder(128, 128, padding=0, kernel_size=z_kernel) # turquoise
        self.down56 = self.bilinear(128, 128, size=self.sizes[6])

        # Level 6
        self.ec661 = self.encoder(128, 128, padding=xy_padding, kernel_size=xy_kernel, n_convs=3) # green
        self.ec662 = self.encoder(128, 128, padding=0, kernel_size=z_kernel) # turquoise
        self.down67 = self.bilinear(128, 256, size=self.sizes[7])

        # Level 7
        self.ec771 = self.encoder(256, 256, padding=xy_padding, kernel_size=xy_kernel, n_convs=3) # green
        self.ec772 = self.encoder(256, 256, padding=0, kernel_size=z_kernel) # turquoise
        self.down78 = self.bilinear(256, bottom_channels, size=self.sizes[8])

        # level 8
        self.ec88 = self.n_linear(bottom_channels, bottom_channels, n_layers=5) # pink arrow

        ## DECODING AND UPSAMPLING##
        self.up87 = self.bilinear(bottom_channels, 256, size=self.sizes2[7])
        self.dc77 = self.decoder(256*2, 256, kernel_size=xy_kernel, padding=xy_padding, n_convs=4)
        self.up76 = self.bilinear(256, 128, size=self.sizes2[6])
        self.dc66 = self.decoder(128*2, 128, kernel_size=xy_kernel, padding=xy_padding, n_convs=4)
        self.up65 = self.bilinear(128, 128, size=self.sizes2[5])
        self.dc55 = self.decoder(128*2, 128, kernel_size=xy_kernel, padding=xy_padding, n_convs=4)
        self.up54 = self.bilinear(128, 64, size=self.sizes2[4])
        self.dc44 = self.decoder(64*2, 64, kernel_size=xy_kernel, padding=xy_padding, n_convs=4)
        self.up43 = self.bilinear(64, 32, size=self.sizes2[3])
        self.dc33 = self.decoder(32*2, 32, kernel_size=xy_kernel, padding=xy_padding, n_convs=3)
        self.up32 = self.bilinear(32, 32, size=self.sizes2[2])
        self.dc22 = self.decoder(32*2, 32, kernel_size=xy_kernel, padding=xy_padding, n_convs=3)
        self.up21 = self.bilinear(32, 32, size=self.sizes2[1])
        self.dc11 = self.decoder(32*2, self.n_classes, kernel_size=xy_kernel, padding=xy_padding, n_convs=3)
        # QUESTION: needed?
        # self.dc10 = self.decoder(32, n_classes, kernel_size=xy_kernel, padding=1, n_convs=4)

    def forward(self, x):
        """
        Define the forward pass of the network
        See supp figure 14 here:
        https://static-content.springer.com/esm/art%3A10.1038%2Fs41591-018-0107-6/MediaObjects/41591_2018_107_MOESM1_ESM.pdf
        """
        def update(tens, means, stds):
            torch.cat((means, get_mean(tens)), dim=1, out=means)
            torch.cat((stds, get_std(tens)), dim=1, out=stds)
        def get_mean(tens):
            tens = tens.view((tens.size(0), -1))
            return torch.mean( tens, dim=1, keepdim=True).detach()
        def get_std(tens):
            tens = tens.view((tens.size(0), -1))
            return torch.std( tens, dim=1, keepdim=True).detach()
        # Collect for info - want it to be tensor(x.size(0), number_of_layers)
        # average over all dimensions apart from input
        means, stds = get_mean(x), get_std(x)

        # l1
        e1 = self.ec_init(x)
        update(e1, means,stds)
        #means, stds = torch.cat((get_mean(e1), means), dim=1), torch.cat((get_std(e1), stds), dim=1)
        
        syn1 = self.ec11(e1)
        update(syn1, means,stds)
        
        e2 = self.down12(syn1)
        update(e2, means,stds)

        # l2
        syn2 = self.ec22(e2)
        update(syn2, means,stds)
        del e1, e2

        e3 = self.down23(syn2)
        update(e3, means,stds)

        # l3
        syn3 = self.ec33(e3)
        update(syn3, means,stds)
        del e3

        e41 = self.down34(syn3)
        update(e41, means,stds)

        # l4
        e42 = self.ec441(e41) 
        update(e42, means,stds)

        syn4 = self.ec442(e42) # right 2 l4 (concat later)
        update(syn4, means,stds)
        del e41, e42
        
        # l5
        e51 = self.down45(syn4) 
        update(e51, means,stds)

        e52 = self.ec551(e51) 
        update(e52, means,stds)

        syn5 = self.ec552(e52)
        update(syn5, means,stds)
        del e51, e52
        
        # l6
        e61 = self.down56(syn5)
        update(e61, means,stds)

        e62 = self.ec661(e61)
        update(e62, means,stds)
        
        syn6 = self.ec662(e62)
        update(syn6, means,stds)
        del e61, e62
        
        # l7
        e71 = self.down67(syn6)
        update(e71, means,stds)
        
        e72 = self.ec771(e71) 
        update(e72, means,stds)

        syn7 = self.ec772(e72)
        update(syn7, means,stds)
        del e71, e72

        # l8
        e_bottom_left = self.down78(syn7)
        update(e_bottom_left, means,stds)
        
        # fc layer
        e_bottom_left = e_bottom_left.view(e_bottom_left.size(0), -1)
        batch_size = e_bottom_left.size()[0]
        e_bottom_right = self.ec88(e_bottom_left)
        
        e_bottom_right = e_bottom_right.view(batch_size, e_bottom_right.size(1), 1,1,1)
        update(e_bottom_right, means,stds)

        ## UPWARD ##

        # l7
        d71 = torch.cat((self.up87(e_bottom_right), self.select_middle(syn7)), dim=1)
        update(d71, means,stds)
        del e_bottom_left, e_bottom_right
        
        d72 = self.dc77(d71) # move right on level 7 (decode)
        update(d72, means,stds)
        del d71, syn7

        # l6
        d61 = torch.cat((self.up76(d72), self.select_middle(syn6)), dim=1)
        update(d61, means,stds)
        del d72, syn6

        d62 = self.dc66(d61)
        update(d62, means,stds)

        # l5
        d51 = torch.cat((self.up65(d62), self.select_middle(syn5)), dim=1)
        update(d51, means,stds)
        del d61, d62, syn5
        
        d52 = self.dc55(d51)
        update(d52, means,stds)

        # l4
        d41 = torch.cat((self.up54(d52), self.select_middle(syn4)), dim=1)
        update(d41, means,stds)
        del d51, d52, syn4

        d42 = self.dc44(d41)
        update(d42, means,stds)

        # l3
        d31 = torch.cat((self.up43(d42), self.select_middle(syn3)), dim=1)
        update(d31, means,stds)
        del d41, d42, syn3

        d32 = self.dc33(d31)
        update(d32, means,stds)

        # l2
        d21 = torch.cat((self.up32(d32), self.select_middle(syn2)), dim=1)
        update(d21, means,stds)
        del d31, d32, syn2

        d22 = self.dc22(d21)
        update(d22, means,stds)

        # l1
        d11 = torch.cat((self.up21(d22), self.select_middle(syn1)), dim=1)
        update(d11, means,stds)
        del d21, d22, syn1

        d12 = self.dc11(d11)
        update(d12, means,stds)
        
        ## QUESTION ##
        # is this right or is there only 1 rightward step at top layer        
        # del d11
        # d0 = self.dc10(d12)
        # return d0

        return means, stds, d12

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, batchnorm=True, n_convs=1):
        """An encoder function, applies conv of kernel size."""
        mods = []
        for n in range(n_convs):

            mods.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            if batchnorm:
                mods.append(nn.BatchNorm3d(out_channels))
            mods.append(nn.ReLU())

        return nn.Sequential(*mods)

    def bilinear(self, in_channels, out_channels, size):
        """Up/Downsample by bilinear interpolation."""

        mods = []

        mods.append(Interpolate(size=size, mode='trilinear'))

        if in_channels != out_channels:
            mods.append(nn.BatchNorm3d(in_channels))
            mods.append(nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=False))
            mods.append(nn.ReLU())

        return nn.Sequential(*mods)

    def n_linear(self, in_channels, out_channels, n_layers=1):
        """A series of n fully connected layers."""

        n_layer_list = []

        for n in range(n_layers):
            n_layer_list.append(nn.Linear(in_channels, out_channels))
            n_layer_list.append(nn.ReLU())

        return nn.Sequential(*n_layer_list)

    # NOTE - currently just reducing number channels at the first step - question for DeepMind
    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=1, output_padding=0, batchnorm=True, bias=False, n_convs=1):
        """An encoder function (upsample)."""
        mods = []
        out = in_channels

        for n in range(n_convs):
            #if n == n_convs - 1:
            if n == 0:
                out = out_channels 
            mods.append(nn.Conv3d(in_channels, out, kernel_size, stride=stride,
                               padding=padding, bias=bias))
            in_channels = out
            # TODO: Check batchnorm?
            if batchnorm:
                mods.append(nn.BatchNorm3d(out))
            mods.append(nn.ReLU())

        layer = nn.Sequential(*mods)

        return layer

    def select_middle(self, syn_large):
        """Returns the middle depth layer of the vector to be passed across."""

        return torch.index_select(syn_large, 2, torch.tensor(syn_large.size(2) // 2).to(self.device))


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = 'trilinear'
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x