"""
Implementation of 3d U-Net
Based on: https://github.com/shiba24/3d-unet
From paper: https://arxiv.org/pdf/1606.06650.pdf

Note, this may also be useful:
https://github.com/jeffkinnison/unet/blob/master/pytorch/unet3d.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    """The network."""

    def __init__(self, in_channel=1, n_classes=15):
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

        self.in_channel = in_channel
        self.n_classes = n_classes

        super(UNet3D, self).__init__()

        ## ENCODING ##
        xy_kernel = (1,3,3)# (3,3,1) # or 1,3,3 ?
        z_kernel = (3,1,1) # (1,1,3) # or 3,1,1 ?
        # Level 1-1
        # QUESTION: this to get 32 channels in layer 1?
        self.ec_init = self.encoder(self.in_channel, 32, padding=0, kernel_size=1, n_convs=1)
        # QUESTION - say used 3x3x1 conv with padding but this increases the z-dimension?
        self.ec11 = self.encoder(32, 32, padding=1, kernel_size=xy_kernel, n_convs=3) # green
        # Level 2->2
        self.ec22 = self.encoder(32, 32, padding=1, kernel_size=xy_kernel, n_convs=3) # green
        # Level 3->3
        self.ec33 = self.encoder(32, 32, padding=1, kernel_size=xy_kernel, n_convs=3) # green
        # Level 4->4
        self.ec441 = self.encoder(64, 64, padding=1, kernel_size=xy_kernel, n_convs=3) # green
        self.ec442 = self.encoder(64, 64, padding=0, kernel_size=z_kernel) # turquoise
        # Level 5->5
        self.ec551 = self.encoder(128, 128, padding=1, kernel_size=xy_kernel, n_convs=3) # green
        self.ec552 = self.encoder(128, 128, padding=0, kernel_size=z_kernel) # turquoise
        # Level 6->6
        self.ec661 = self.encoder(128, 128, padding=1, kernel_size=xy_kernel, n_convs=3) # green
        self.ec662 = self.encoder(128, 128, padding=0, kernel_size=z_kernel) # turquoise
        # Level 7->7
        self.ec771 = self.encoder(256, 256, padding=1, kernel_size=xy_kernel, n_convs=3) # green
        self.ec772 = self.encoder(256, 256, padding=0, kernel_size=z_kernel) # turquoise
        # level 8->8
        self.ec88TEMP = self.n_linear(256, 256, n_layers=5) # pink arrow
        self.ec88 = self.n_linear(4096, 4096, n_layers=5) # pink arrow

        ## DOWNSAMPLING ##
        # Defined in forward

        ## DECODING AND UPSAMPLING[moved into forward] ##
        self.dc77 = self.decoder(256*2, 256, kernel_size=xy_kernel, padding=1, n_convs=4)
        self.dc66 = self.decoder(128*2, 128, kernel_size=xy_kernel, padding=1, n_convs=4)
        self.dc55 = self.decoder(128*2, 128, kernel_size=xy_kernel, padding=1, n_convs=4)
        self.dc44 = self.decoder(64*2, 64, kernel_size=xy_kernel, padding=1, n_convs=4)
        self.dc33 = self.decoder(32*2, 32, kernel_size=xy_kernel, padding=1, n_convs=4)
        self.dc22 = self.decoder(32*2, 32, kernel_size=xy_kernel, padding=1, n_convs=4)
        self.dc11 = self.decoder(32*2, 32, kernel_size=xy_kernel, padding=1, n_convs=4)
        # Into classes - maybe this is the same as above?
        self.dc10 = self.decoder(32, n_classes, kernel_size=xy_kernel, padding=1, n_convs=4)

    def forward(self, x):
        """
        Define the forward pass of the network
        See supp figure 14 here:
        https://static-content.springer.com/esm/art%3A10.1038%2Fs41591-018-0107-6/MediaObjects/41591_2018_107_MOESM1_ESM.pdf
        """
        # l1
        print("input", x.shape)
        e1 = self.ec_init(x)
        print("init", e1.shape)
        syn1 = self.ec11(e1) # init right - l1
        print("syn1", syn1.shape)
        e2 = self.bilinear(syn1, 32, 32, scale_factor=0.5) # l1-2
        print("e2", e2.shape)
        # l2
        syn2 = self.ec22(e2) # right l2 (concat later)
        print("syn2", syn2.shape)
        del e1, e2
        e3 = self.bilinear(syn2, 32, 32, scale_factor=0.5) # l2-3
        print("e3", e3.shape)
        # l3
        syn3 = self.ec33(e3) # right l3 (concat later)
        print("syn3", syn3.shape)
        del e3 # delete
        e41 = self.bilinear(syn3, 32, 64, scale_factor=0.5) # l3-l4
        print("e41", e41.shape)

        # l4
        e42 = self.ec441(e41) # right 1 l4
        syn4 = self.ec442(e42) # right 2 l4 (concat later)
        print("syn4", syn4.shape)
        del e41, e42
        e51 = self.bilinear(syn4, 64, 128, scale_factor=0.5) # l4-l5
        print("e51", e51.shape)
        # l5
        e52 = self.ec551(e51) # right 1
        syn5 = self.ec552(e52) # right 2
        print("syn5", syn5.shape)
        del e51, e52
        
        e61 = self.bilinear(syn5, 128, 128, scale_factor=0.5) # l5-l6
        print("e61", e61.shape)
        # l6
        e62 = self.ec661(e61) # right 1
        syn6 = self.ec662(e62) # right 2
        print("syn6", syn6.shape)
        del e61, e62
        e71 = self.bilinear(syn6, 128, 256, scale_factor=0.5) #l6-7
        print("e71", e71.shape)
        # l7
        e72 = self.ec771(e71) # right 1 (green)
        syn7 = self.ec772(e72) # right 2 (turq)
        print("syn7", syn7.shape)
        del e71, e72

        #e_bottom_left = self.bilinear(syn7, 256, 4092, scale_factor=0.125) # l7-l8
        e_bottom_leftTEMP = self.bilinear(syn7, 256, 256, scale_factor=1) # l7-l8
        print("e_b_l", e_bottom_leftTEMP.shape)

        # l8 - the very bottom most encoded
        #e_bottom_right = self.ec88(e_bottom_left)
        e_bottom_right = self.ec88TEMP(e_bottom_leftTEMP)
        print("e_b_r", e_bottom_right.shape)

        ## DECODE ##
        # QUESTION - check this is a simple cat - says "copy and stack"
        d71TEMP = torch.cat((self.bilinear(e_bottom_right, 256, 256, scale_factor=1), syn7)) # concat on level 7
        print("shape 7 cat", d71TEMP.shape)
        #d71 = torch.cat((self.bilinear(e_bottom_right, 4092, 256, scale_factor=8), syn7)) # concat on level 7
        del e_bottom_left, e_bottom_right
        d72 = self.dc77(d71TEMP) # move right on level 7 (decode)
        print("shape 7 decoded", d72.shape)
        #d72 = self.dc77(d71)
        del d71TEMP, syn7
        #del d71, syn7

        # TODO - finish
        d61 = torch.cat((self.bilinear(d72, 256, 128, scale_factor=2), syn6))
        d62 = self.dc66(d61)
        del d72, d61, syn6

        d51 = torch.cat((self.bilinear(d62, 128, 128, scale_factor=2), syn5))
        d52 = self.dc55(d51)
        del d62, d51, syn5

        d41 = torch.cat((self.bilinear(d52, 128, 64, scale_factor=2), syn4))
        d42 = self.dc44(d41)
        del d41, d52, syn4

        d31 = torch.cat((self.bilinear(d42, 64, 32, scale_factor=2), syn3))
        d32 = self.dc33(d31)
        del d31, d42, syn3

        d21 = torch.cat((self.bilinear(d32, 32, 32, scale_factor=2), syn2))
        d22 = self.dc22(d21)
        del d21, d32, syn2

        d11 = torch.cat((self.bilinear(d22, 32, 32, scale_factor=2), syn1))
        d12 = self.dc11(d11)
        del d11, d22, syn1

        # QUESTION
        # is this right or is there only 1 rightward step at top layer
        d0 = self.dc10(d12)
        return d0

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, batchnorm=True, n_convs=1):
        """An encoder function, applies conv of kernel size."""
        mods = []
        for n in range(n_convs):

            mods.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            # TODO: Check batchnorm?
            if batchnorm:
                mods.append(nn.BatchNorm3d(out_channels))

            # TODO - check if activation is ReLU and attached every conv?
            mods.append(nn.ReLU())

        layer = nn.Sequential(*mods)

        return layer

    # QUESTION - figure out align_corners / trilinear/bilinear
    def bilinear(self, x, in_channels, out_channels, scale_factor):
        """Up/Downsample by bilinear interpolation."""

        # TODO - for each z-layer in x - instead of trilinear

        y = F.interpolate(x, scale_factor=scale_factor,
                             mode='trilinear', align_corners=False)
        if in_channels != out_channels:
            expand = self.encoder(in_channels, out_channels, padding=0, kernel_size=1, n_convs=1)
            y = expand(y)
        return y

    def n_linear(self, in_channels, out_channels, n_layers=1):
        """A series of n fully connected layers."""

        n_layer_list = []

        for n in range(n_layers):
            n_layer_list.append(nn.Linear(in_channels, out_channels))
            n_layer_list.append(nn.ReLU())

        layer = nn.Sequential(*n_layer_list)

        return layer

    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=1, output_padding=0, bias=False, n_convs=1):
        """An encoder function (upsample)."""

        mods = []

        for n in range(n_convs):
            mods.append(nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias))
            mods.append(nn.ReLU())

        layer = nn.Sequential(*mods)

        return layer
