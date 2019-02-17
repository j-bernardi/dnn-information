"""
Implementation of 3d U-Net
Based on: https://github.com/shiba24/3d-unet
From paper: https://arxiv.org/pdf/1606.06650.pdf

Note, this may also be useful:
https://github.com/jeffkinnison/unet/blob/master/pytorch/unet3d.py
"""

"""
TODO???
Third, we introduced one extra residual connection within each block of layers, 
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
        #self.sizes2 = [(0,0,0), (9, 448, 512), (9, 224, 256), (9, 112, 128), (7, 56, 64), 
        #              (5, 28, 32), (3, 14, 16), (1, 7, 8), (1, 1, 1)]

        # Smaller down-sizes:
        self.sizes = [(0,0,0), (9, 112, 128), (9, 112, 128), (9, 112, 128), (9, 56, 64), 
                      (7, 28, 32), (5, 14, 16), (3, 7, 8), (1, 1, 1)]
        # Back up:
        self.sizes2 = [(0,0,0), (9, 112, 128), (9, 112, 128), (9, 112, 128), (7, 56, 64), 
                      (5, 28, 32), (3, 14, 16), (1, 7, 8), (1, 1, 1)]

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

        # Level 1-1
        # QUESTION: this to get 32 channels in layer 1?
        self.ec_init = self.encoder(self.in_channel, 32, padding=0, kernel_size=1, n_convs=1)
        self.ec11 = self.encoder(32, 32, padding=xy_padding, kernel_size=xy_kernel, n_convs=3)
        self.down12 = self.bilinear(32, 32, size=self.sizes[2])

        # Level 2->2
        self.ec22 = self.encoder(32, 32, padding=xy_padding, kernel_size=xy_kernel, n_convs=3)
        self.down23 = self.bilinear(32, 32, size=self.sizes[3])

        # Level 3->3
        self.ec33 = self.encoder(32, 32, padding=xy_padding, kernel_size=xy_kernel, n_convs=3)
        self.down34 = self.bilinear(32, 64, size=self.sizes[4])

        # Level 4->4
        self.ec441 = self.encoder(64, 64, padding=xy_padding, kernel_size=xy_kernel, n_convs=3) # green
        self.ec442 = self.encoder(64, 64, padding=0, kernel_size=z_kernel) # turquoise
        self.down45 = self.bilinear(64, 128, size=self.sizes[5])

        # Level 5->5
        self.ec551 = self.encoder(128, 128, padding=xy_padding, kernel_size=xy_kernel, n_convs=3) # green
        self.ec552 = self.encoder(128, 128, padding=0, kernel_size=z_kernel) # turquoise
        self.down56 = self.bilinear(128, 128, size=self.sizes[6])

        # Level 6->6
        self.ec661 = self.encoder(128, 128, padding=xy_padding, kernel_size=xy_kernel, n_convs=3) # green
        self.ec662 = self.encoder(128, 128, padding=0, kernel_size=z_kernel) # turquoise
        self.down67 = self.bilinear(128, 256, size=self.sizes[7])

        # Level 7->7
        self.ec771 = self.encoder(256, 256, padding=xy_padding, kernel_size=xy_kernel, n_convs=3) # green
        self.ec772 = self.encoder(256, 256, padding=0, kernel_size=z_kernel) # turquoise
        self.down78 = self.bilinear(256, bottom_channels, size=self.sizes[8])

        # level 8->8
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
        # l1
        #print("INIT SIZE", torch.cuda.max_memory_allocated())
        #print("L1")
        #print("input", x.shape)
        e1 = self.ec_init(x)
        #print("init", e1.shape)
        syn1 = self.ec11(e1) # init right - l1
        #print("syn1", syn1.shape)
        #print("L2")
        e2 = self.down12(syn1) # l1-2
        #print("e2", e2.shape)
        # l2
        syn2 = self.ec22(e2) # right l2 (concat later)
        #print("syn2", syn2.shape)
        del e1, e2
        e3 = self.down23(syn2) # l2-3
        #print("L3")
        #print("e3", e3.shape)
        # l3
        syn3 = self.ec33(e3) # right l3 (concat later)
        #print("syn3", syn3.shape)
        del e3 # delete
        #print("L4")
        e41 = self.down34(syn3) # l3-l4
        #print("e41", e41.shape)

        # l4
        e42 = self.ec441(e41) # right 1 l4
        #print("e42", e42.shape)        
        syn4 = self.ec442(e42) # right 2 l4 (concat later)
        #print("syn4", syn4.shape)
        del e41, e42
        #print("L5")
        e51 = self.down45(syn4) # l4-l5
        #print("e51", e51.shape)
        # l5
        e52 = self.ec551(e51) # right 1
        #print("e52", e52.shape)
        syn5 = self.ec552(e52) # right 2
        #print("syn5", syn5.shape)
        del e51, e52
        #print("L6")
        e61 = self.down56(syn5) # l5-l6
        #print("e61", e61.shape)
        
        # l6
        e62 = self.ec661(e61) # right 1
        #print("e62", e62.shape)
        syn6 = self.ec662(e62) # right 2
        #print("syn6", syn6.shape)
        del e61, e62
        #print("L7")
        e71 = self.down67(syn6) #l6-7
        #print("e71", e71.shape)
        
        # l7
        e72 = self.ec771(e71) # right 1 (green)
        #print("e72", e72.shape)
        syn7 = self.ec772(e72) # right 2 (turq)
        #print("syn7", syn7.shape)
        del e71, e72

        #print("L8")

        #e_bottom_left = self.bilinear(syn7, 256, 4092, size=self.sizes[8]) # l7-l8
        e_bottom_left = self.down78(syn7) # l7-l8
        #print("e_b_l", e_bottom_left.shape)

        # l8 - the very bottom most encoded
        e_bottom_left = e_bottom_left.view(e_bottom_left.size(0), -1)
        batch_size = e_bottom_left.size()[0]
        
        e_bottom_right = self.ec88(e_bottom_left)
        
        e_bottom_right = e_bottom_right.view(batch_size, e_bottom_right.size(1), 1,1,1)
        #print("e_b_r", e_bottom_right.shape)

        #print("SIZE BEFORE DEL", torch.cuda.max_memory_allocated())
        #if torch.cuda.is_available():
        #    torch.cuda.empty_cache()
            #print("SIZE AFTER DEL", torch.cuda.max_memory_allocated())

        ## DECODE ##
        #print("TO CONCAT:")
        #print("Shape1", self.bilinear(e_bottom_right, 4096, 256, size=self.sizes2[7]).shape)
        #print("Shape1", self.bilinear(e_bottom_right, 256, 256, size=self.sizes2[7]).shape)
        #print("syn7  ", syn7.shape)
        # QUESTION - check this is a simple cat - says "copy and stack"
        #d71 = torch.cat((self.bilinear(e_bottom_right, 4096, 256, size=self.sizes2[7]), syn7), dim=1) # concat on level 7
        d71 = torch.cat((self.up87(e_bottom_right), syn7), dim=1) # concat on level 7
        #print("d71 (post cat)", d71.shape)
        del e_bottom_left, e_bottom_right
        d72 = self.dc77(d71) # move right on level 7 (decode)
        #print("d72 (decoded)", d72.shape)
        del d71, syn7

        # TODO - finish
        d61 = torch.cat((self.up76(d72), syn6), dim=1)
        del d72, syn6
        d62 = self.dc66(d61)

        d51 = torch.cat((self.up65(d62), syn5), dim=1)
        del d61, d62, syn5
        d52 = self.dc55(d51)

        d41 = torch.cat((self.up54(d52), syn4), dim=1)
        del d51, d52, syn4
        d42 = self.dc44(d41)

        d31 = torch.cat((self.up43(d42), syn3), dim=1)
        del d41, d42, syn3
        d32 = self.dc33(d31)

        d21 = torch.cat((self.up32(d32), syn2), dim=1)
        del d31, d32, syn2
        d22 = self.dc22(d21)

        d11 = torch.cat((self.up21(d22), syn1), dim=1)
        del d21, d22, syn1
        d12 = self.dc11(d11)
        
        """
        del d11
        # QUESTION
        # is this right or is there only 1 rightward step at top layer
        d0 = self.dc10(d12)
        return d0
        """

        return d12

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, batchnorm=True, n_convs=1):
        """An encoder function, applies conv of kernel size."""
        mods = []
        for n in range(n_convs):

            mods.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            if batchnorm:
                mods.append(nn.BatchNorm3d(out_channels))
            mods.append(nn.ReLU())

        return nn.Sequential(*mods)

    # QUESTION - figure out align_corners
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

    # QUESTION - currently just reducing number channels at the final step - or should it be sequential?
    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=1, output_padding=0, batchnorm=True, bias=False, n_convs=1):
        """An encoder function (upsample)."""
        mods = []
        out = in_channels

        for n in range(n_convs):
            if n == n_convs - 1:
                out = out_channels 
            mods.append(nn.Conv3d(in_channels, out, kernel_size, stride=stride,
                               padding=padding, bias=bias))
            # TODO: Check batchnorm?
            if batchnorm:
                mods.append(nn.BatchNorm3d(out))
            mods.append(nn.ReLU())

        layer = nn.Sequential(*mods)

        return layer


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = 'trilinear'
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x