"""
Implementation of 3d U-Net
Based on: https://github.com/shiba24/3d-unet
From paper: https://arxiv.org/pdf/1606.06650.pdf

Note, this may also be useful:
https://github.com/jeffkinnison/unet/blob/master/pytorch/unet3d.py
"""
import torch
import torch.nn as nn

class UNet3D(nn.Module):
    """The network."""

    def __init__(self, in_channel=32, n_classes=15):
        """
        Input: (default) 32-channel 448x512x9 voxels image
        Output: estimated probability over the 15 (default) classes
        (for each of the 448x512x1 output voxels)
        """

        self.in_channel = in_channel
        self.n_classes = n_classes

        super(UNet3D, self).__init__()

        # Define the encoders
        """
        # see https://arxiv.org/pdf/1606.06650.pdf
        # 1-1, 1
        self.ec0 = self.encoder(self.in_channel, 32, bias=False, batchnorm=False)
        # 1-1, 2
        self.ec1 = self.encoder(32, 64, bias=False, batchnorm=False)
        # 2-2, 1
        self.ec2 = self.encoder(64, 64, bias=False, batchnorm=False)
        # 2-2, 2
        self.ec3 = self.encoder(64, 128, bias=False, batchnorm=False)
        # 3-3, 1
        self.ec4 = self.encoder(128, 128, bias=False, batchnorm=False)
        # 3-3, 2
        self.ec5 = self.encoder(128, 256, bias=False, batchnorm=False)
        # 4-4, 1
        self.ec6 = self.encoder(256, 256, bias=False, batchnorm=False)
        # 4-4, 2
        self.ec7 = self.encoder(256, 512, bias=False, batchnorm=False)
        """
        # TODO: no padding to input, bilinear downsampling
        # Level 1-1
        self.ec11 = self.encoder(self.in_channel, 32) # green
        # Level 2->2
        self.ec22 = self.encoder(32, 32) # green REDEFINE
        # Level 3->3
        self.ec33 = self.encoder(32, 32) # green
        # Level 4->4
        self.ec441 = self.encoder(64, 64) # green
        self.ec442 = self.encoder(64, 64) # turquoise REDEFINE
        # Level 5->5
        self.ec551 = self.encoder(128, 128) # green
        self.ec552 = self.encoder(128, 128) # turquoise
        # Level 6->6
        self.ec661 = self.encoder(128, 128) # green
        self.ec662 = self.encoder(128, 128) # turquoise
        # Level 7->7
        self.ec771 = self.encoder(256, 256) # green
        self.ec772 = self.encoder(256, 256) # turquoise
        self.ec773 = self.encoder(256, 256*2) # blue
        # level 8->8
        self.ec88 = self.encoder(4096, 4096) # pink arrow

        # TODO: point 2, do through bilinear interpolation
        # replace max pooling and up conv - shouldn't be needed?
        # these become the downwards?
        # TODO - define the convs (3x3x1, 1x1x3) (think these go in encoder, decoder?)
        # self.down0, down1, down2
        """
        self.pool0 = nn.MaxPool3d(2) # l1-l2
        self.pool1 = nn.MaxPool3d(2) # l2-l3
        self.pool2 = nn.MaxPool3d(2) # l3-l4 (bottom)
        """
        self.down12 = # TODO # l1-l2
        self.down23 = # TODO # l2-l3
        self.down34 = # TODO # l3-l4
        self.down45 = # TODO # l4-l5
        self.down56 = # TODO # l5-l6
        self.down67 = # TODO # l6-l7
        self.down78 = # TODO # l7-l8

        # define upward decoders
        """
        # See: https://arxiv.org/pdf/1606.06650.pdf
        self.dc9 = self.decoder(512, 512, kernel_size=2, stride=2, bias=False)
        self.dc8 = self.decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc7 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6 = self.decoder(256, 256, kernel_size=2, stride=2, bias=False)
        self.dc5 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3 = self.decoder(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc2 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc0 = self.decoder(64, n_classes, kernel_size=1, stride=1, bias=False)
        """
         # TODO - figure out kernel size, stride and bias
         # all green on diagram
        # Level 8->8
        self.dc88 = self.decoder(4096, 4096)
        # Level 8->7
        self.dc87 = self.decoder(4096, 256*2, kernel_size=2, stride=2, bias=False)
        # Level 7->7
        self.dc77 = self.decoder(256*2, 256)
        # Level 7->6
        self.dc76 = self.decoder(256, 128*2, kernel_size=3, stride=1, padding=1, bias=False)
        # Level 6->6
        self.dc66 = self.decoder(128*2, 128)
        # Level 6->5
        self.dc65 = self.decoder(128, 128*2, kernel_size=3, stride=1, padding=1, bias=False)
        # Level 5->5
        self.dc55 = self.decoder(128*2, 128)
        # Level 5->4
        self.dc54 = self.decoder(128, 128*2, kernel_size=2, stride=2, bias=False)
        # level 4->4
        self.dc44 = self.decoder(128*2, 128)
        # Level 4->3
        self.dc43 = self.decoder(128, 64*2, kernel_size=3, stride=1, padding=1, bias=False)
        # level 3->3
        self.dc33 = self.decoder(64*2, 64)
        # Level 3->2
        self.dc32 = self.decoder(64, 32*2, kernel_size=3, stride=1, padding=1, bias=False)
        # Level 2->2
        self.dc22 = self.decoder(32*2, 32)
        # Level 2->1
        self.dc21 = self.decoder(32, 32*2, kernel_size=2, stride=2, bias=False)
        # Level 1->1
        self.dc11 = self.decoder(32*2, 32)
        # Into classes
        self.dc10 = self.decoder(32, n_classes)

    def forward(self, e1):
        """
        # start at l1
        e0 = self.ec0(x) # right
        syn0 = self.ec1(e0) # right (l1)
        e1 = self.pool0(syn0) # down (to l2)
        e2 = self.ec2(e1) # right
        syn1 = self.ec3(e2) # right
        del e0, e1, e2
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
        e72 = self.ec771(e71) # right 1
        syn7 = self.ec772(e72) # right 2
        del e71, e72

        # l8 - the very bottom most encoded
        e_bottom_right = self.ec88(syn7) # can delete later

        ## DECODE ##

        """
        ## DECODING ##

        d9 = torch.cat((self.dc9(e7), syn2)) # concat - adding level 3
        del e7, syn2

        d8 = self.dc8(d9) # right (compress level 3)
        d7 = self.dc7(d8) # right (normal level 3)
        del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1)) # concat - adding to level 2
        del d7, syn1

        d5 = self.dc5(d6) # right (l2)
        d4 = self.dc4(d5) # right (l2)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0)) # concat - adding to level 1
        del d4, syn0

        d2 = self.dc2(d3) # right
        d1 = self.dc1(d2) # right
        del d3, d2
        """

        d0 = self.dc0(d1) # final right - output
        return d0

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True, batchnorm=False):
        """An encoder function (downsample)."""

        # TODO: Check batchnorm and if activation is ReLU?
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())

        return layer


    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=1, output_padding=0, bias=True):
        """An encoder function (upsample)."""
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer
