import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from model.BusmodderNet import BusmodderNet
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from torch.nn import Linear, Conv1d, BatchNorm1d, MaxPool1d, Dropout
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax

import sys
sys.path.append("../DCRNN")
from lib.utils import calculate_normalized_laplacian
from misc.data_loader import load_network, adjacency_matrix

class CNN(BusmodderNet):

    def __init__(self):
        super(CNN, self).__init__()
        self.road_network = load_network()
        self.adj_mat = adjacency_matrix(self.road_network)
        self.laplacian = calculate_normalized_laplacian(self.adj_mat)
        #out_dim = (input_dim - filter_dim + 2padding) / stride + 1
        channels = 1
        kernel_size = 10
        stride = 0
        padding = 0
        #TODO: should be read instead of hardcoded
        feature_size = 194
        self.conv_1 = Conv1d(in_channels=1,
                            out_channels=channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
        #
        #self.conv_out_height1 = compute_conv_dim(height)
        #self.conv_out_width1 = compute_conv_dim(width)
        #self.conv1_in_features1 = channels * height * width
        
        max_pool_1_size = 2
        self.max_pool_1 = MaxPool1d(kernel_size=max_pool_1_size, stride=2)
        #self.max_pool_out_height1 = compute_max_pool_dim(self.conv_out_height1)
        #self.max_pool_out_width1 = compute_max_pool_dim(self.conv_out_width1)
        
        #self.conv_2 = Conv2d(in_channels=num_filters_conv1,
        #                    out_channels=num_filters_conv1,
        #                    kernel_size=kernel_size_conv1,
        #                    stride=stride_conv1,
        #                    padding=padding_conv1)
        
        
        #self.conv_out_height2 = compute_conv_dim(self.max_pool_out_height1)
        #self.conv_out_width2 = compute_conv_dim(self.max_pool_out_width1)
        #self.conv1_in_features2 = channels * self.conv_out_height1 * self.conv_out_width1
        
        #self.max_pool_2 = MaxPool2d(kernel_size=max_pool_1_size, stride=2)
        #self.max_pool_out_height2 = compute_max_pool_dim(self.conv_out_height2)
        #self.max_pool_out_width2 = compute_max_pool_dim(self.conv_out_width2)
        
        # add dropout to network
        #self.dropout = Dropout2d(p=0.5)
        self.l1_in_features = channels * feature_size
        #self.l1_in_features = channels * height * width
        
        #self.l_1 = Linear(in_features=self.l1_in_features, 
        #                  out_features=num_l1,
        #                  bias=True)
        self.l_out = Linear(in_features=self.l1_in_features, 
                            out_features=feature_size,
                            bias=False)
    
    def forward(self, x): # x.size() = [batch, channel, height, width]
        x = relu(self.conv_1(x))
        #x = self.dropout(relu(self.conv_1(x)))
        #x = self.max_pool_1(x)
        #x = self.dropout(relu(self.conv_2(x)))
        #x = self.max_pool_2(x)
        # torch.Tensor.view: http://pytorch.org/docs/master/tensors.html?highlight=view#torch.Tensor.view
        #   Returns a new tensor with the same data as the self tensor,
        #   but of a different size.
        # the size -1 is inferred from other dimensions 
        
        
        #x = self.dropout(relu(self.l_1(x)))
        #x = relu(self.l_1(x))
        return relu(self.l_out(x), dim=1)

