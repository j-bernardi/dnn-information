import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: We also used a small amount (0.05) of label-smoothing regularization
## TODO - input dimensions? 15 channels in and 14 out?
## and added some (1 × 10−5) weight decay.

class CNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(2, 2, 2, 3),
                 in_channels=15, bn_size=4, drop_rate=0, n_classes=14):

        super(CNet, self).__init__()

        # First convolution
        # QUESTION: is it 3-in_channels or 15 (and in_chanels is different?)
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(3, in_channels, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(in_channels)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # dense blocks
        num_features = in_channels
        for i, num_layers in enumerate(block_config):

            # Make the block
            block = _DenseBlock(num_layers, num_features, bn_size,
                                growth_rate, drop_rate)
            # Update
            self.features.add_module('denseblock%d' % (i+1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features//2)
                self.features.add_module('transition%d' % (i+1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # final linear level
        self.classifier = nn.Linear(num_features, num_classes)

        # QUESTION: What does this do?
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1)).view(features.size(0), -1)
        # out = self.classifier(out) # not included in mine?
        return out

class _DenseBlock(nn.Sequential):
    """
    Applies the xy and z convolutions in a block. Details:
        https://arxiv.org/pdf/1608.06993.pdf
        https://github.com/liuzhuang13/DenseNet
        https://pytorch.org/docs/master/_modules/torchvision/models/densenet.html
    Return the convolved vector and the output size (channels)
    """

    def __init__(self, n_z, in_channels, bn_size, growth_rate, drop_rate):
        for i in range(n_z):
            layer = self._DenseLayer(in_channels + i * growth_rate,
                                      growth_rate, bn_size, int(n_xy/n_z),
                                      drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _DenseLayer(nn.Sequential):
    """
    Add a specific layer.
        https://arxiv.org/pdf/1608.06993.pdf
        https://github.com/liuzhuang13/DenseNet
        https://pytorch.org/docs/master/_modules/torchvision/models/densenet.html
    Return the convolved vector and the output size (channels)
    """
    def __init__(self, in_channels, growth_rate, bn_size, m_xy, drop_rate=0):
        super(_DenseLayer, self).__init__()
        m, pd, ks = 0, 0, (3,3,1)
        for while m <= m_xy:
            if m == m_xy:
                pd = 1
                ks = (1,1,3)
            self.add_module('norm%d' % (m+1), nn.BatchNorm3d(in_channels)),
            self.add_module('relu%d' % (m+1), nn.ReLU(inplace=True)),
            self.add_module('conv' % (m+1), nn.Conv3d(in_channels,
                bn_size*growth_rate, kernel_size=ks, stride=1, bias=False,
                padding=pd))
            m += 1
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        # halve the number of modules
        self.add_module('pool', nn.MaxPool3d(kernel_size=2, stride=2))

"""
# OLD version of CNet init then forward...

        # TODO - check pooling handles edges correctly with padding
        self.max_pool = nn.MaxPool3d(2)

        # l1 green, 2 xy convs 3-5 channels
        input_size = in_channels
        self.conv1, input_size = dense_conv(input_size, n_xy=2, n_z=0)
        # max pool

        # l2, 4 xy convs, 2 z convs 5-
        self.conv2, input_size = dense_conv(input_size, n_xy=4, n_z=2)
        # max pool

        # l3, pink 1x1x1 conv
        self.conv31 = nn.Conv3d(input_size, input_size/4, 1)
        # l3, green 4 xy convs, 2 z-convs
        self.conv32, input_size = dense_conv(input_size/4, n_xy=4, n_z=2)

        # l4 pink
        self.conv41 = self.conv31 = nn.Conv3d(input_size, input_size/2, 1)
        # l4 green 4 xy 2 z
        self.conv42, input_size = _DenseBlock(input_size/2, n_xy=4, n_z=2)

        # l5 pink 1
        self.conv51 = self.conv31 = nn.Conv3d(input_size, input_size/1.5, 1)
        #l5 green
        self.conv52, input_size = dense_conv(input_size/1.5, n_xy=6, n_z=3)
        # l5 pink 2
        self.conv53 = self.conv31 = nn.Conv3d(input_size, n_classes, 1)

        # final stage - average the whole space for each channel
        # Does this bring it down into a single dimension tensor, length n_classes?
        self.global_pool = nn.AvgPool3d((18,21,5))

    def forward(self, x):

        x = self.max_pool(self.conv1(x)) #l1-2
        x = self.max_pool(self.conv2(x)) #l2-3
        x = self.max_pool(self.conv32(self.conv31(x))) # l3-4
        x = self.max_pool(self.conv42(self.conv41(x))) #l4-5
        x = self.conv53(self.conv52(self.conv51(x))) # l5-end
        return F.softmax(self.global_pool(x))
"""
