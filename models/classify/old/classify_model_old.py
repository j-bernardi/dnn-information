import torch, math
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

## TODO - growth rate stuff and bn_size - don't know anything about this

class CNet(nn.Module):

    def __init__(self, growth_rate=32, in_channels=15, bn_size=4, drop_rate=0, n_classes=14):

        super(CNet, self).__init__()

        # Make first up/down sampling

        num_features = in_channels
        self.features = nn.Sequential(OrderedDict([]))
        # QUESTION - do we need an initial convolution?

        # BLOCK 1 (l1-2)
        print("block1")
        print("into dense block", num_features)
        block1 = _DenseBlock(num_features, bn_size, growth_rate, drop_rate, num_xy=2, num_z=0)
        num_features = num_features + 2 * growth_rate
        print("Out of dense block", num_features)
        print("into trans", num_features)
        trans1 = _Transition(num_features, num_features, conv=False)
        num_features = num_features
        print("out of trans1", num_features)
        self.features.add_module('denseblock1', block1)
        self.features.add_module('transition1', trans1)

        # BLOCK 2 (l2-3) - 4x xy, 2x z, then transition(conv=True)
        print("block2")
        print("into dense block", num_features)
        # TODO - this one fails?
        block2 = _DenseBlock(num_features, bn_size, growth_rate, drop_rate, num_xy=4, num_z=2)
        num_features = num_features + 4 * 2 * growth_rate
        print("Out of dense block", num_features)
        trans2 = _Transition(num_features, num_features//4)
        num_features = num_features // 4
        print("out of trans", num_features)
        self.features.add_module('denseblock2', block2)
        self.features.add_module('transition2', trans2)

        # BLOCK 3 (l3-4) - 4x xy, 2x z, then transition(conv=True)
        print("block3")
        print("into dense block", num_features)
        block3 = _DenseBlock(num_features, bn_size, growth_rate, drop_rate, num_xy=4, num_z=2)
        num_features = num_features + 4 * 2 * growth_rate
        print("Out of dense block", num_features)
        trans3 = _Transition(num_features, int(math.floor(num_features / 1.5)))
        num_features = int(math.floor(num_features / 1.5))
        print("out of trans", num_features)
        self.features.add_module('denseblock3', block3)
        self.features.add_module('transition3', trans3)

        # BLOCK 4 (l4-5) - 4x xy, 2x z, then transition(conv=True)
        print("block4")
        print("into dense block", num_features)
        block4 = _DenseBlock(num_features, bn_size, growth_rate, drop_rate, num_xy=4, num_z=2)
        num_features = num_features + 4 * 2 * growth_rate
        print("Out of dense block", num_features)
        trans4 = _Transition(num_features, num_features//15)
        num_features = num_features // 15
        print("out of trans", num_features)
        self.features.add_module('denseblock4', block4)
        self.features.add_module('transition4', trans4)

        # BLOCK 5 (l5-6) - 6x xy, 3x z then final_transition(conv=True(before), average pool)
        print("block5")
        print("into dense block", num_features)
        block5 = _DenseBlock(num_features, bn_size, growth_rate, drop_rate, num_xy=6, num_z=3)
        num_features = num_features + 6 * 3 * growth_rate
        print("Out of dense block", num_features)
        self.features.add_module('denseblock5', block5)

        # GROWTH RATE: lth layer has k_0 + k * (l-1) for each function Hlproducing k feature maps
        # k is the growth rate

        # Final convolution
        print("final")
        self.features.add_module('final_norm', nn.BatchNorm3d(num_features))
        self.features.add_module('relu', nn.ReLU(inplace=True))
        self.features.add_module('final_conv', nn.Conv3d(
            num_features, n_classes, kernel_size=1)
        )
        print("final features", num_features)

        # QUESTION: What does this do?
        # 'Official init from torch repo'
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        # 1. downsample (or upsample) to input size
        # TEMP size
        x = self.bilinear(x, x.size(1), x.size(1), size=(23, 60, 70))
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1)).view(features.size(0), -1)
        return out

    def bilinear(self, x, in_channels, out_channels, size=(43,300,350)):
        """Up/Downsample by bilinear interpolation."""

        # TODO - for each z-layer in x - instead of trilinear
        # or bilinear per sheet?
        y = F.interpolate(x, size=size,
                             mode='trilinear', align_corners=False)
        if in_channels != out_channels:
            expand = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                #nn.BatchNorm3d(out_channels),
                nn.ReLU()
            )
            expand = expand.to(self.device)
            y = expand(y)
        return y

# TODO - fix the input sizes
class _DenseBlock(nn.Sequential):
    """
    Applies the xy and z convolutions in a block. Details:
        https://arxiv.org/pdf/1608.06993.pdf
        https://github.com/liuzhuang13/DenseNet
        https://pytorch.org/docs/master/_modules/torchvision/models/densenet.html
    Return the convolved vector and the output size (channels)
    """

    def __init__(self, in_channels, bn_size, growth_rate, drop_rate, num_xy, num_z, x_before_z=2):
        # number of z convs applied
        super(_DenseBlock, self).__init__()
        print("Creating DenseBlock")
        if num_z == 0:
            # just do x_before_z xs
            for i in range(x_before_z):
                layer = _DenseLayer(in_channels + i * growth_rate,
                                          growth_rate, bn_size, drop_rate, ks=(1,3,3))
                self.add_module('denselayer%d' % (i + 1), layer)

        else:
            for nz in range(num_z):
                # Add the xy layers
                for i in range(x_before_z):
                    layer = _DenseLayer(in_channels + i * (nz) * growth_rate,
                        growth_rate, bn_size, drop_rate, ks=(1,3,3))
                    self.add_module('denselayer%d' % (nz)*(i + 1), layer)

                # Add the z layer
                layer = _DenseLayer(in_channels + (i+1) * (nz) * growth_rate,
                                      growth_rate, bn_size, drop_rate, ks=(3,1,1))
                self.add_module('denselayer%d' % (nz * (i + 1)), layer)

# TODO - add padding to stop x y z channels decreasing
class _DenseLayer(nn.Sequential):
    """
    Add a specific layer.
        https://arxiv.org/pdf/1608.06993.pdf
        https://github.com/liuzhuang13/DenseNet
        https://pytorch.org/docs/master/_modules/torchvision/models/densenet.html
    Return the convolved vector and the output size (channels)
    """
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate=0, ks=(1,3,3)):
        super(_DenseLayer, self).__init__()
        
        if ks == (1,3,3):
            pd = (0,1,1)
        elif ks == (3,1,1):
            pd = (1,0,0)
        
        self.ks = ks # TEMP

        self.add_module('norm1', nn.BatchNorm3d(in_channels)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(in_channels, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),

        self.add_module('norm2', nn.BatchNorm3d(bn_size*growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size*growth_rate, growth_rate,
                        kernel_size=ks, stride=1, padding=pd, bias=False))
        
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        print("Dense Layer completing ks", self.ks)
        print("x shape", x.shape)
        print("features shape", new_features.shape)
        return torch.cat([x, new_features], 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, conv=True):
        super(_Transition, self).__init__()
        print("Creating transition")
        # max pool - halve the number of modules
        self.add_module('pool', nn.MaxPool3d(kernel_size=2, stride=2))
        if conv:
            self.add_module('norm', nn.BatchNorm3d(num_input_features))
            self.add_module('relu', nn.ReLU(inplace=True))
            self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                              kernel_size=1, stride=1, bias=False))
