import torch, math
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

## TODO - growth rate stuff and bn_size - don't know anything about this

class CNet(nn.Module):

    def __init__(self, growth_rate=2, in_channels=15, bn_size=4, drop_rate=0, n_classes=14, device="find"):

        super(CNet, self).__init__()

        self.xy_kernel, self.z_kernel = (1,3,3), (3,1,1)
        self.xy_padding, self.z_padding = (0,1,1), (1,0,0)

        # The number of channels appended by a layer
        self.growth_rate = growth_rate
        self.in_channels = in_channels
        
        # Bottleneck size - e.g. bottlneck layer has growth-rate * bottleneck
        self.bn_size = bn_size

        self.in_channel = in_channels
        self.n_classes = n_classes
        
        if device == "find":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else: 
            self.device = torch.device(device)

        # 1. downsample (or upsample) to input size
        # TEMP size
        sz = (43, 300, 350)
        sz_small = (23, 60, 70)
        self.first = self.bilinear(in_channels, in_channels, size=sz)

        # Block 1
        self.a11 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a12 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate

        self.down12 = self.downsample(in_channels, in_channels, conv=False)

        # Block 2
        self.a21 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a22 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a23 = self.z_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a24 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a25 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a26 = self.z_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate

        # TODO - check right number of out channels?? Pass size to right size
        self.down23 = self.downsample(in_channels, in_channels // 4, conv=True)
        in_channels = in_channels // 4

        # Block 3
        self.a31 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a32 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a33 = self.z_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a34 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a35 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a36 = self.z_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate

        # Check out channels
        self.down34 = self.downsample(in_channels, in_channels // 2, conv=True)
        in_channels = in_channels // 2

        # Block 4
        self.a41 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a42 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a43 = self.z_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a44 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a45 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a46 = self.z_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate

        # Check out channels
        self.down45 = self.downsample(in_channels, int(math.floor(in_channels / 1.5)), conv=True)
        in_channels = int(math.floor(in_channels / 1.5))

        # Block 5
        self.a51 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a52 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a53 = self.z_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a54 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a55 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a56 = self.z_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a57 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a58 = self.xy_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate
        self.a59 = self.z_conv(in_channels, bn_size, growth_rate)
        in_channels += growth_rate

        self.last_step = self.conv_once(in_channels, n_classes)

    def forward(self, x):

        y = self.first(x)

        # 1 2 x - y convs at initial size
        y = torch.cat((y, self.a11(y)), 1)
        y = torch.cat((y, self.a12(y)), 1)

        # Max pooling to 150x175x43
        y = self.down12(y)

        # 2 x-y convs - out channels is bn size then concat
        y = torch.cat((y, self.a21(y)), 1)
        y = torch.cat((y, self.a22(y)), 1)
        # 1 z conv
        y = torch.cat((y, self.a23(y)), 1)
        # 2 x-y convs - out channels is bn size then concat
        y = torch.cat((y, self.a24(y)), 1)
        y = torch.cat((y, self.a25(y)), 1)
        # 1 z conv
        y = torch.cat((y, self.a26(y)), 1)

        # max pooling to 75x87x21 w conv
        y = self.down23(y)

        # 2xy convs
        y = torch.cat((y, self.a31(y)), 1)
        y = torch.cat((y, self.a32(y)), 1)
        # 1 z conv
        y = torch.cat((y, self.a33(y)), 1)
        # 2xy convs
        y = torch.cat((y, self.a34(y)), 1)
        y = torch.cat((y, self.a35(y)), 1)
        # 1 z conv
        y = torch.cat((y, self.a36(y)), 1)

        # max pooling to 37x43x10 w conv
        y = self.down34(y)

        # 2xy convs
        y = torch.cat((y, self.a41(y)), 1)
        y = torch.cat((y, self.a42(y)), 1)
        # 1 z conv
        y = torch.cat((y, self.a43(y)), 1)
        # 2xy convs
        y = torch.cat((y, self.a44(y)), 1)
        y = torch.cat((y, self.a45(y)), 1)
        # 1 z conv
        y = torch.cat((y, self.a46(y)), 1)

        # max pooling to 18x21x5
        y = self.down45(y)

        # 2xy convs
        y = torch.cat((y, self.a51(y)), 1)
        y = torch.cat((y, self.a52(y)), 1)
        # 1 z conv
        y = torch.cat((y, self.a53(y)), 1)
        # 2xy convs
        y = torch.cat((y, self.a54(y)), 1)
        y = torch.cat((y, self.a55(y)), 1)
        # 1 z conv
        y = torch.cat((y, self.a56(y)), 1)
        # 2xy convs
        y = torch.cat((y, self.a57(y)), 1)
        y = torch.cat((y, self.a58(y)), 1)
        # 1 z conv
        y = torch.cat((y, self.a59(y)), 1)

        # TODO - check size out and check gradient being passed
        y = self.last_step(y)

        out = F.relu(y, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1)).view(out.size(0), -1)

        #print("out", out.size())

        return out

    def bilinear(self, in_channels, out_channels, size=(43,300,350)):
        """Up/Downsample by bilinear interpolation."""

        mods = []

        mods.append(Interpolate(size=size, mode='trilinear'))

        if in_channels != out_channels:
            mods.append(nn.BatchNorm3d(in_channels))
            mods.append(nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=False))
            mods.append(nn.ReLU())

        return nn.Sequential(*mods)

    def xy_conv(self, in_channels, bn_size, growth_rate):
        
        mods = []
        ## Bottleneck layer - improve computational efficiency
        mods.append(nn.BatchNorm3d(in_channels))
        mods.append(nn.ReLU(inplace=True))
        mods.append(nn.Conv3d(in_channels, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False))

        mods.append(nn.BatchNorm3d(bn_size*growth_rate))
        mods.append(nn.ReLU(inplace=True))
        mods.append(nn.Conv3d(bn_size*growth_rate, growth_rate,
                        kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False))        

        return nn.Sequential(*mods)

    def z_conv(self, in_channels, bn_size, growth_rate):
        
        mods = []
        ## QUESTION - this block necessary?
        mods.append(nn.BatchNorm3d(in_channels))
        mods.append(nn.ReLU(inplace=True))
        mods.append(nn.Conv3d(in_channels, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False))

        ## Bottleneck layer - improve computational efficiency
        mods.append(nn.BatchNorm3d(bn_size*growth_rate))
        mods.append(nn.ReLU(inplace=True))
        mods.append(nn.Conv3d(bn_size*growth_rate, growth_rate,
                        kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False))        
        
        return nn.Sequential(*mods)

    def downsample(self, in_channels, out_channels, conv=True):

        mods = []

        """
        # In paper but not in DeepMind?
        # conv layer
        mods.append(nn.BatchNorm3d(in_channels))
        mods.append(nn.Conv3d(in_channels, in_channels, kernel_size=1,padding=0,bias=False))
        """

        # max pool - halve the number of modules
        mods.append(nn.MaxPool3d(kernel_size=2, stride=2))
        if conv:
            mods.append(self.conv_once(in_channels, out_channels))

        return nn.Sequential(*mods)

    def conv_once(self, in_channels, out_channels):

        mods = []

        mods.append(nn.BatchNorm3d(in_channels))
        mods.append(nn.ReLU(inplace=True))
        mods.append(nn.Conv3d(in_channels, out_channels,
                    kernel_size=1, stride=1, bias=False))

        return nn.Sequential(*mods)

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = 'trilinear'
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x