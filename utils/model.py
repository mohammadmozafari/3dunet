import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, input_dim, output_dim, initial_filters=16, depth=4, dropout=0):
        # input_dim: number of input channles. 1 in our case.
        # output dim: number of output channels. 3 in our case. (since we have 3 labels)
        # initial_filter: kernel_size of the first conv. usually 16 or 32.
        # depth: depth of the U in unet. `image_slices` should be divisible by (2 ** depth)
        # dropout: the possibility of zeroing a node out.
        
        super(UNet3D, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initial_filters = initial_filters
        self.depth = depth
        self.dropout = dropout
        activation = nn.ReLU(inplace=True)
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.trans = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Down
        for i in range(depth):
            in_dim = self.input_dim
            if i > 0:
                in_dim = self.initial_filters * 2 ** (i - 1)
            out_dim = self.initial_filters * 2 ** i
            self.downs.append(self.double_conv_block(in_dim, out_dim, activation))
            self.pools.append(self.max_pool())
            
        
        # Bridge
        self.bridge = self.double_conv_block(
            self.initial_filters * 2 ** (depth - 1),
            self.initial_filters * 2 ** depth,
            activation
        )
        
        # Dropout
        self.dropouts.append(self.dropout_layer())
        
        # Up
        for i in range(depth):
            trans_in_out_dim = self.initial_filters * 2 ** (depth - i)
            self.trans.append(self.conv_transpose_block(trans_in_out_dim, trans_in_out_dim, activation))
            
            up_in_dim = self.initial_filters * (2 ** (depth - i) + 2 ** (depth - i - 1))
            up_out_dim = self.initial_filters * 2 ** (depth - i - 1)
            self.ups.append(self.double_conv_block(up_in_dim, up_out_dim, activation))
            
        # Dropout
        self.dropouts.append(self.dropout_layer())
        
        # Output
        self.out = self.prediction_mask(initial_filters, self.output_dim)
        
    def single_conv_block(self, input_dim, output_dim, activation):
        return nn.Sequential(
            nn.Conv3d(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(output_dim),
            activation
        )

    def double_conv_block(self, input_dim, output_dim, activation):
        return nn.Sequential(
            self.single_conv_block(input_dim, output_dim, activation),
            #nn.Dropout(p=self.dropout),
            self.single_conv_block(output_dim, output_dim, activation)
        )

    def conv_transpose_block(self, input_dim, output_dim, activation):
        return nn.Sequential(
            nn.ConvTranspose3d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(output_dim),
            activation,
            #nn.Dropout(p=self.dropout)
        )

    def prediction_mask(self, input_dim, output_dim):
        # the activation is considered to in the loss.
        return nn.Conv3d(input_dim, output_dim, kernel_size=1, stride=1, padding=0) 

    def max_pool(self):
        return nn.MaxPool3d(kernel_size=2, stride=2, padding=0) 
    
    def dropout_layer(self):
        return nn.Dropout(p=self.dropout)
        
    def forward(self, x):
        downs = []
        ups = []
        pools = []
        trans = []
        concats = []
        dropouts = []
        
        # Down
        for i in range(self.depth):
            inp = x
            if i > 0:
                inp = pools[-1]
            downs.append(self.downs[i](inp))
            pools.append(self.pools[i](downs[-1]))
        
        # Bridge
        bridge = self.bridge(pools[-1])
        
        # Dropout
        dropouts.append(self.dropouts[0](bridge))
        
        # Up
        for i in range(self.depth):
            inp = dropouts[-1]
            if i > 0:
                inp = ups[-1]
            trans.append(self.trans[i](inp))
            concats.append(torch.cat([trans[-1], downs[self.depth - i - 1]], dim=1))
            ups.append(self.ups[i](concats[-1]))
        
        # Dropout
        dropouts.append(self.dropouts[1](ups[-1]))
        
        # Output
        out = self.out(dropouts[-1])
        return out