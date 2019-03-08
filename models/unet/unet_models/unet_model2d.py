"""
Implementation of 2d U-Net
Based on:
https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

Code based on:
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""

import torch, gc
import numpy as np
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
        def update(mod, num_layers, append=False):
            """Updates the number of layers and returns the number """
            if append:
                self.info_layers_numbers.append(num_layers)
            return len(list(mod.children()))

        super(UNet2D, self).__init__()

        self.in_channel = in_channel
        self.n_classes = n_classes

        if device == "find":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else: 
            self.device = torch.device(device)

        # The reps per epoch
        self.representations_per_epochs = []
        # A list holding all the reps from this epoch
        self.current_representations = None
        # The indices of the layers to calculate info for (e.g. the convs)
        self.info_layers_numbers = []

        ## Define the model ##

        num_layers = 0

        ## ENCODING ##

        # QUESTION: this to get 32 channels in layer 1?
        self.ec1 = self.encoder(1, 64)
        num_layers += update(self.ec1, num_layers, append=True)
        self.down12 = Interpolate(max_pool=True)
        num_layers += update(self.down12, num_layers)
        
        # l2
        self.ec2 = self.encoder(64, 128)
        num_layers += update(self.ec2, num_layers, append=True)
        self.down23 = Interpolate(max_pool=True)
        num_layers += update(self.down23, num_layers)
        
        # l3
        self.ec3 = self.encoder(128, 256)
        num_layers += update(self.ec3, num_layers, append=True)
        self.down34 = Interpolate(max_pool=True)
        num_layers += update(self.down34, num_layers)
        
        self.ec4 = self.encoder(256, 512)
        num_layers += update(self.ec4, num_layers, append=True)
        self.down45 = Interpolate(max_pool=True)
        num_layers += update(self.down45, num_layers)
        
        ## DECODING ##

        # TODO - consider fc here to compensate for 9 classes (not 2)
        
        self.ec5 = self.encoder(512, 1024)
        num_layers += update(self.ec5, num_layers, append=True)
        self.up54 = self.decoder(1024, 512)
        num_layers += update(self.up54, num_layers)
        
        self.dc4 = self.encoder(1024, 512)
        num_layers += update(self.dc4, num_layers, append=True)
        self.up43 = self.decoder(512, 256)
        num_layers += update(self.up43, num_layers)
        
        self.dc3 = self.encoder(512, 256)
        num_layers += update(self.dc3, num_layers, append=True)
        self.up32 = self.decoder(256, 128)
        num_layers += update(self.up32, num_layers)
        
        self.dc2 = self.encoder(256, 128)
        num_layers += update(self.dc2, num_layers, append=True)
        self.up21 = self.decoder(128, 64)
        num_layers += update(self.up21, num_layers)
        
        self.dc1 = self.encoder(128, 64)
        num_layers += update(self.dc1, num_layers, append=True)

        ### TODO - check no activation required ###
        self.final_step = nn.Sequential(
            nn.Conv2d(64, self.n_classes, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.n_classes)
        )

        ## Create the list ###
        num_layers += update(self.final_step, num_layers, append=True)
        
        
        # Check tracking correct layers
        print("Info layers numbers", self.info_layers_numbers)
        """
        for i in range(num_layers):
            print(i)
            if i in  self.info_layers_numbers:
                print("*** TRACKING INFO ***")
        """
    
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
        syn1 = self.ec1(x)
        self.add_info(0, syn1.detach().cpu().numpy())
        del x

        # L2
        e2 = self.down12(syn1)
        syn2 = self.ec2(e2)
        self.add_info(1, syn2.detach().cpu().numpy())
        del e2

        # L3
        e3 = self.down23(syn2)
        syn3 = self.ec3(e3)
        self.add_info(2, syn3.detach().cpu().numpy())
        del e3

        # L4
        e4 = self.down34(syn3)
        syn4 = self.ec4(e4)
        self.add_info(3, syn4.detach().cpu().numpy())
        del e4

        # L5
        e51 = self.down45(syn4)
        e52 = self.ec5(e51)
        self.add_info(4, e52.detach().cpu().numpy())
        del e51

        ## UPWARD ##

        up = self.up54(e52)
        d41 = torch.cat(
            (up, self.crop_to(syn4, up)), dim=1)
        del syn4, e52
        
        d42 = self.dc4(d41)
        self.add_info(5, d42.detach().cpu().numpy())
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
        self.add_info(6, d32.detach().cpu().numpy())
        del d31

        # L2
        up = self.up32(d32)
        d21 = torch.cat(
            (up, self.crop_to(syn2, up)), dim=1)
        del syn2, d32

        d22 = self.dc2(d21)
        self.add_info(7, d22.detach().cpu().numpy())
        del d21

        # L1
        up = self.up21(d22)
        d11 = torch.cat(
            (up, self.crop_to(syn1, up)), dim=1)
        del syn1, d22
        d12 = self.dc1(d11)
        self.add_info(8, d12.detach().cpu().numpy())
        del d11
        
        out = self.final_step(d12)
        self.add_info(9, out.detach().cpu().numpy())

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

    def next_epoch(self):
        """Appends the current reps to the epoch reps, resets."""

        # Save the epoch - list of representations in this epoch per layer being saved
        self.representations_per_epochs.append(self.current_representations)

        # Empty out current reps
        self.reset() 

    def add_info(self, layer_index, representations):
        """Add this layer output for the info analysis."""

        # List holding all the representations of the epoch, per level
        if self.current_representations[layer_index] is None:
            self.current_representations[layer_index] = representations

        else:

            # To concatenate
            assert self.current_representations[layer_index].shape[1:] == representations.shape[1:]
            
            self.current_representations[layer_index] = np.concatenate([self.current_representations[layer_index],
                                                                   representations], axis=0)
            
            #print("result", self.current_representations[layer_index].shape)

    def reset(self):
        """Resets the current [epoch] representations."""
        self.current_representations = [None for _ in range(len(self.info_layers_numbers))]

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
