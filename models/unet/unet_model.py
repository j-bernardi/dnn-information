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

    def __init__(self, in_channel=32, n_classes=15):
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

        # Level 1-1
        self.ec11 = self.encoder(self.in_channel, 32, padding=1, kernel_size=(3,3,1), n_convs=3) # green
        # Level 2->2
        self.ec22 = self.encoder(32, 32, padding=1, kernel_size=(3,3,1), n_convs=3) # green
        # Level 3->3
        self.ec33 = self.encoder(32, 32, padding=1, kernel_size=(3,3,1), n_convs=3) # green
        # Level 4->4
        self.ec441 = self.encoder(64, 64, padding=1, kernel_size=(3,3,1), n_convs=3) # green
        self.ec442 = self.encoder(64, 64, padding=0, kernel_size=(1,1,3)) # turquoise
        # Level 5->5
        self.ec551 = self.encoder(128, 128, padding=1, kernel_size=(3,3,1), n_convs=3) # green
        self.ec552 = self.encoder(128, 128, padding=0, kernel_size=(1,1,3)) # turquoise
        # Level 6->6
        self.ec661 = self.encoder(128, 128, padding=1, kernel_size=(3,3,1), n_convs=3) # green
        self.ec662 = self.encoder(128, 128, padding=0, kernel_size=(1,1,3)) # turquoise
        # Level 7->7
        self.ec771 = self.encoder(256, 256, padding=1, kernel_size=(3,3,1), n_convs=3) # green
        self.ec772 = self.encoder(256, 256, padding=0, kernel_size=(1,1,3)) # turquoise
        # level 8->8
        self.ec88 = self.n_linear(4096, 4096, n_layers=5) # pink arrow

        ## DOWNSAMPLING ##

        self.down12 = self.bilinear(32, scale_factor=1) # TODO # l1-l2
        self.down23 = self.bilinear(32, scale_factor=1) # TODO # l2-l3
        self.down34 = self.bilinear(32, scale_factor=2) # TODO # l3-l4
        self.down45 = self.bilinear(64, scale_factor=2) # TODO # l4-l5
        self.down56 = self.bilinear(128, scale_factor=1) # TODO # l5-l6
        self.down67 = self.bilinear(128, scale_factor=2) # TODO # l6-l7
        self.down78 = self.bilinear(256, scale_factor=16) # TODO # l7-l8

        ## DECODING AND UPSAMPLING ##
        # Level 8->7
        self.dc87 = self.bilinear(4096, scale_factor=0.0625)
        # Level 7->7
        self.dc77 = self.decoder(256*2, 256, kernel_size=(3,3,1), padding=1, n_convs=4)
        # Level 7->6
        self.dc76 = self.bilinear(256, scale_factor=0.5)
        # Level 6->6
        self.dc66 = self.decoder(128*2, 128, kernel_size=(3,3,1), padding=1, n_convs=4)
        # Level 6->5
        self.dc65 = self.bilinear(128, scale_factor=1)
        # Level 5->5
        self.dc55 = self.decoder(128*2, 128, kernel_size=(3,3,1), padding=1, n_convs=4)
        # Level 5->4
        self.dc54 = self.bilinear_up(128, scale_factor=0.5)
        # level 4->4
        self.dc44 = self.decoder(64*2, 64, kernel_size=(3,3,1), padding=1, n_convs=4)
        # Level 4->3
        self.dc43 = self.bilinear(64, scale_factor=0.5)
        # level 3->3
        self.dc33 = self.decoder(32*2, 32, kernel_size=(3,3,1), padding=1, n_convs=4)
        # Level 3->2
        self.dc32 = self.bilinear(32, scale_factor=1)
        # Level 2->2
        self.dc22 = self.decoder(32*2, 32, kernel_size=(3,3,1), padding=1, n_convs=4)
        # Level 2->1
        self.dc21 = self.bilinear(32, scale_factor=1)
        # Level 1->1
        self.dc11 = self.decoder(32*2, 32, kernel_size=(3,3,1), padding=1, n_convs=4)

        # Into classes - maybe this is the same as above?
        self.dc10 = self.decoder(32, n_classes, kernel_size=(3,3,1), padding=1, n_convs=4)

    def forward(self, e1):
        """
        Define the forward pass of the network
        See supp figure 14 here:
        https://static-content.springer.com/esm/art%3A10.1038%2Fs41591-018-0107-6/MediaObjects/41591_2018_107_MOESM1_ESM.pdf
        """
        # l1
        syn1 = self.ec11(e1) # init right - l1
        e2 = self.down12(syn1) # down to l2

        # l2
        syn2 = self.ec22(e1) # right l2 (concat later)
        e3 = self.down23(syn2)

        # l3
        syn3 = self.ec33(e3) # right l3 (concat later)
        del e2, e3 # delete
        e41 = self.down34(syn3) # down to l4

        # l4
        e42 = self.ec441(e41) # right 1 l4
        syn4 = self.ec442(e42) # right 2 l4 (concat later)
        del e41, e42
        e51 = self.down45(syn4) # down to l5

        # l5
        e52 = self.ec551(e51) # right 1
        syn5 = self.ec552(e52) # right 2
        del e51, e52
        e61 = self.down56(syn5) # down to l6

        # l6
        e62 = self.ec661(e61) # right 1
        syn6 = self.ec662(e62) # right 2
        del e61, e62
        e71 = self.down67(syn6) #down to 7

        # l7
        e72 = self.ec771(e71) # right 1 (green)
        syn7 = self.ec772(e72) # right 2 (turq)
        del e71, e72

        # l8 - the very bottom most encoded
        e_bottom_right = self.ec88(syn7)

        ## DECODE ##
        # QUESTION - check this is a simple cat - says "copy and stack"
        d71 = torch.cat((self.dc87(e_bottom_right), syn7)) # concat on level 7
        d72 = self.dc77(d71) # move right on level 7 (decode)
        del e_bottom_right, d71, syn7

        # TODO - finish
        d61 = torch.cat((self.dc76(d72), syn6))
        d62 = self.dc66(d61)
        del d72, d61, syn6

        d51 = torch.cat((self.dc65(d62), syn5))
        d52 = self.dc55(d51)
        del d62, d51, syn5

        d41 = torch.cat((self.dc54(d52), syn4))
        d42 = self.dc44(d41)
        del d41, d52, syn4

        d31 = torch.cat((self.dc43(d42), syn3))
        d32 = self.dc33(d31)
        del d31, d42, syn3

        d21 = torch.cat((self.dc32(d32), syn2))
        d22 = self.dc22(d21)
        del d21, d32, syn2

        d11 = torch.cat((self.dc21(d22), syn1))
        d12 = self.dc11(d11)
        del d11, d22, syn1

        # QUESTION
        # is this right or is there only 1 rightward step at top layer
        d0 = self.dc10(d12)
        return d0

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True, batchnorm=True, n_convs=1):
        """An encoder function, applies conv of kernel size."""
        mods = []
        for n in range(n_convs):

            mods.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))

            # TODO: Check batchnorm?
            if batchnorm:
                mods.append(nn.BatchNorm2d(out_channels))

            # TODO - check if activation is ReLU and attached every conv?
            mods.append(F.ReLU())

        layer = nn.Sequential(*mods)

        return layer

    # QUESTION - figure out align_corners
    def bilinear(self, in_channels, scale_factor):
        """Up/Downsample by bilinear interpolation."""

        return F.interpolate(input_size=in_channels,
                        scale_factor=scale_factor, mode='bilinear',
                        align_corners=False)

    def n_linear(self, in_channels, out_channels, n_layers=1):
        """A series of n fully connected layers."""

        n_layer_list = []

        for n in n_layers:
            n_layer_list.append(nn.Linear(in_channels, out_channels))

        layer = nn.Sequential(*n_layer_list)

        return layer

    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=1, output_padding=0, bias=True, n_convs=1):
        """An encoder function (upsample)."""

        mods = []

        for n in n_convs:
            mods.append(nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias))
            mods.append(F.ReLU())

        layer = nn.Sequential(*mods)

        return layer
