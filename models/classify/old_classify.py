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
