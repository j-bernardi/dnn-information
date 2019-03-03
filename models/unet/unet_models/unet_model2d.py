"""
Implementation of 2d U-Net
Based on:
https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

Code based on:
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""

import torch, gc
import torch.nn as nn
import torch.nn.functional as F

class UNet2D(nn.Module):
    """The network."""

    def __init__(self, in_channel=1, n_classes=9, device="find"):
        """
        Archtecture modelled from:
            https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
        Default:
            Input: (default) 32-channel 448x512x9 voxels image
            Output: estimated probability over the 15 (default) classes
                (for each of the 448x512x1 output voxels)
        """

        self.in_channel = in_channel
        self.n_classes = n_classes

        if device == "find":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else: 
            self.device = torch.device(device)

        super(UNet2D, self).__init__()

        list_of_layers = []

        ## ENCODING ##

        # QUESTION: this to get 32 channels in layer 1?
        self.ec1 = self.encoder(1, 64)
        list_of_layers.extend(list(self.ec1.children()))
        self.down12 = Interpolate(max_pool=True)
        list_of_layers.extend(list(self.down12.children()))
        
        self.ec2 = self.encoder(64, 128)
        list_of_layers.extend(list(self.ec2.children()))
        self.down23 = Interpolate(max_pool=True)
        list_of_layers.extend(list(self.down23.children()))
        
        self.ec3 = self.encoder(128, 256)
        list_of_layers.extend(list(self.ec3.children()))
        self.down34 = Interpolate(max_pool=True)
        list_of_layers.extend(list(self.down34.children()))
        
        self.ec4 = self.encoder(256, 512)
        list_of_layers.extend(list(self.ec4.children()))
        self.down45 = Interpolate(max_pool=True)
        list_of_layers.extend(list(self.down45.children()))
        
        ## DECODING ##

        # TODO - consider fc here to compensate for 9 classes (not 2)
        
        self.ec5 = self.encoder(512, 1024)
        list_of_layers.extend(list(self.ec5.children()))
        self.up54 = self.decoder(1024, 512)
        list_of_layers.extend(list(self.up54.children()))
        
        self.dc4 = self.encoder(1024, 512)
        list_of_layers.extend(list(self.dc4.children()))
        self.up43 = self.decoder(512, 256)
        list_of_layers.extend(list(self.up43.children()))
        
        self.dc3 = self.encoder(512, 256)
        list_of_layers.extend(list(self.dc3.children()))
        self.up32 = self.decoder(256, 128)
        list_of_layers.extend(list(self.up32.children()))
        
        self.dc2 = self.encoder(256, 128)
        list_of_layers.extend(list(self.dc2.children()))
        self.up21 = self.decoder(128, 64)
        list_of_layers.extend(list(self.up21.children()))
        
        self.dc1 = self.encoder(128, 64)
        list_of_layers.extend(list(self.dc1.children()))

        self.final_step = nn.Sequential(
            nn.Conv2d(64, self.n_classes, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.n_classes)
        )
        list_of_layers.extend(list(self.final_step.children()))

        # Combine into one sequential item for info
        self.layers = nn.Sequential(*list_of_layers)
    
    def forward(self, x):
        """
        Define the forward pass of the network
        https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
        """

        ## UPSAMPLE ##
        
        # TEMP - later, consider just making sure downsample always even
        # Not needed if padding=1
        #x = F.interpolate(x, size=(572, 572))
        
        ## DOWN ##

        # L1
        #print("x", x.shape)
        syn1 = self.ec1(x)
        #print("s1", syn1.shape)
        del x

        # L2
        e2 = self.down12(syn1)
        #print("e2", e2.shape)
        syn2 = self.ec2(e2)
        #print("s2", syn2.shape)
        del e2

        # L3
        e3 = self.down23(syn2)
        #print("e3", e3.shape)
        syn3 = self.ec3(e3)
        #print("syn3", syn3.shape)
        del e3

        # L4
        e4 = self.down34(syn3)
        syn4 = self.ec4(e4) 
        del e4

        # L5
        e51 = self.down45(syn4)
        e52 = self.ec5(e51)
        del e51

        ## UPWARD ##

        up = self.up54(e52)
        d41 = torch.cat(
            (up, self.crop_to(syn4, up)), dim=1)
        del syn4, e52
        
        d42 = self.dc4(d41)
        del d41

        # L3
        up = self.up43(d42)
        # FAILS when 512 
        #print("up", up.shape)
        #print("syn", syn3.shape) # odd 
        d31 = torch.cat(
            (up, self.crop_to(syn3, up)), dim=1)
        del syn3, d42
        
        d32 = self.dc3(d31)
        del d31

        # L2
        up = self.up32(d32)
        d21 = torch.cat(
            (up, self.crop_to(syn2, up)), dim=1)
        del syn2, d32

        d22 = self.dc2(d21)
        del d21

        # L1
        up = self.up21(d22)
        d11 = torch.cat(
            (up, self.crop_to(syn1, up)), dim=1)
        del syn1, d22
        d12 = self.dc1(d11)
        del d11
        
        out = self.final_step(d12)

        return out
        
    def crop_to(self, big, small):
        """Crops 4d tensor x to size of tensor y in last 2 dimensions."""

        #print("small size", small.size(2), "big size", big.size(2))
        cropY = (big.size(2) - small.size(2)) // 2
        cropX = (big.size(3) - small.size(3)) // 2

        #print("taking big[", cropY, ":", big.size(2) - cropY, "]")
        #print("E.g.", big.size(2) - 2*cropY)

        big = big[:, :, 
              cropY : big.size(2) - cropY,
              cropX : big.size(3) - cropX]

        return big

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True, n_convs=2):
        """An encoder function, applies conv of kernel size."""
        
        mods = []

        for n in range(n_convs):

            # Add convolution
            mods.append(nn.Conv2d(in_channels, out_channels, 
                                  kernel_size=kernel_size, stride=stride, 
                                  padding=padding, bias=bias))
            
            in_channels = out_channels
            
            if batchnorm:
                mods.append(nn.BatchNorm2d(out_channels))
            
            mods.append(nn.ReLU())

        return nn.Sequential(*mods)

    def decoder(self, in_channels, out_channels, bilinear=False):
        """An encoder function (upsample)."""
        mods = []

        if bilinear:
            return Interpolate(max_pool=False, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

class Interpolate(nn.Module):
    """Interpolates up or down through various means."""

    def __init__(self, max_pool=False, scale_factor=0.5, kernel_size=2, mode='bilinear', align_corners=False):
        
        super(Interpolate, self).__init__()
        
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.align_corners = align_corners
        self.max_pool = max_pool
        self.mode = mode

        # Downward max pool, kernel_size 2
        if max_pool:
            self.interp = nn.functional.max_pool2d
            self.args = {"kernel_size" : self.kernel_size}
        
        # Upsample using interpolation, scale factor = 0.5
        else:
            self.interp = nn.functional.interpolate
            self.args = {"scale_factor" : self.scale_factor, 
                         "mode" : self.mode, 
                         "align_corners" : self.align_corners}
     
    def forward(self, x):
        
        x = self.interp(x, **self.args)
        
        return x
