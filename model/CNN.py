import torch
from BaseNetwork import BaseNetwork
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from torch.nn.modules import Module
from torch.nn import Module, Linear, ReLU, Sequential
from BaseNetwork import BaseNetwork
from torch.optim import Adam
import sys
sys.path.append("../DCRNN")
sys.path.append("../misc")
from lib.utils import calculate_normalized_laplacian
from misc.data_loader import load_network, adjacency_matrix
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from MoviaBusDataset import MoviaBusDataset
class ConvolutionLayer(Module):
    def __init__(self, laplacian, n_out, timesteps, bias=None):
        super(ConvolutionLayer,self).__init__()
        self.n_roads = laplacian.shape[0]
        self.laplacian = laplacian
        #self.bias = bias
        #self.l1  = Linear(n_in * self.n_nodes, n_out * self.n_nodes,bias = False)
        #self.weight = Parameter(torch.cuda.FloatTensor(n_in,self.n_nodes, n_out))
        #self.weight = Parameter(torch.cuda.FloatTensor(n_out,1,self.n_nodes, n_in))
        self.theta = Parameter(torch.cuda.FloatTensor(n_out, 1, self.n_roads))
        self.omega = Parameter(torch.cuda.FloatTensor(timesteps))
        if bias:
            raise "Not implimented"
            self.bias = Parameter(torch.cuda.FloatTensor(self.n_roads,n_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.theta.size(1))
        self.theta.data.uniform_(-stdv, stdv)
        stdv_omega = 1. / math.sqrt(self.omega.size(0))
        self.omega.data.uniform_(-stdv_omega, stdv_omega)
        if self.bias is not None:
            raise "Not implimented"
            self.bias.data.uniform_(-stdv, stdv)

    
    def forward(self,x):
        y = self.laplacian.matmul(x)
        y = y.unsqueeze(1)
        #print(x.shape)
        #print("y " + str(y.shape))
        #y = y.reshape(shape[0],shape[1]*shape[2])
        #print(y.shape)
        #print("theta " +str(self.theta.shape))
        #y = self.l1(y)
        y = self.theta.matmul(y)
        #print("y " + str(y.shape))
        y = y.matmul(self.omega)
        #print("y " + str(y.shape))
        #y = y.reshape(shape)
        #y = y.squeeze(2)
        #print("y " + str(y.shape))
        return y


class CNN(BaseNetwork):
    def __init__(self, previous_timesteps):
        super(CNN,self).__init__()
        self.road_network = load_network(MoviaBusDataset.hack_filters)
        self.adj_mat = adjacency_matrix(self.road_network)
        #first order approx
        self.n_roads = self.adj_mat.shape[0]
        sparse_mat = calculate_normalized_laplacian(self.adj_mat + torch.eye(self.n_roads))
        self.laplacian = torch.cuda.FloatTensor(sparse_mat.todense()) if torch.cuda.is_available()  else torch.FloatTensor(sparse_mat.todense())#self.convl1 = ConvolutionLayer(self.laplacian,(previous_timesteps+1),1)
        self.relu = ReLU()
        self.CNN = Sequential(
            ConvolutionLayer(self.laplacian, self.n_roads, previous_timesteps+1),
            ReLU(),
            ConvolutionLayer(self.laplacian, self.n_roads,1),
            ReLU(),
            ConvolutionLayer(self.laplacian, self.n_roads,1),
            #ReLU(),
            #ConvolutionLayer(self.laplacian, self.n_roads,1),
            #ReLU(),
            #ConvolutionLayer(self.laplacian,num_hidden, 1),
            #Linear(num_hidden*(previous_timesteps+1), self.n_nodes),
            #Linear(num_hidden * (previous_timesteps+1), 1),
        )
        
    
    
    def forward(self,x):
        """
        x : [batch_size, prev_timesteps, num_roads]
        
        """
        #Transpose input, such that the previous time steps are the last dimension
        x = x.transpose(2,1)
        
        predictions = []
        for _ in range(self.max_timestep):
            #Run the input through the network
            y = self.CNN(x).squeeze()
            #y = self.convl1(x)
            #y = self.relu(y)
            #Append the prediction to the list of predictions
            predictions.append(y[:,:self.num_roads])

            #remove oldest timestep
            x = x[:,:,1:]
            #unsqueeze output so its size is [batch_size, num_roads, timesteps]
            #print(y.shape)
            y = y.unsqueeze(2)
            #append the new prediction to the input
            x = torch.cat((x,y),dim=2)
        return torch.stack(predictions,1)

    
    
class ConvolutionLayer_diffusion(Module):
    def __init__(self, laplacian, n_out, timesteps, max_diffusion_step, bias=None):
        super(ConvolutionLayer_diffusion,self).__init__()
        self.max_diffusion_step = max_diffusion_step
        self.n_roads = laplacian.shape[0]
        self.laplacian = laplacian
        #self.bias = bias
        self.theta = Parameter(torch.cuda.FloatTensor(n_out, 1, self.n_roads))  # original
        #self.theta = Parameter(torch.cuda.FloatTensor(1, self.n_roads))
        self.omega = Parameter(torch.cuda.FloatTensor(timesteps))
        
        if bias:
            raise "Not implimented"
            self.bias = Parameter(torch.cuda.FloatTensor(self.n_roads,n_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.theta.size(1))
        self.theta.data.uniform_(-stdv, stdv)
        stdv_omega = 1. / math.sqrt(self.omega.size(0))
        self.omega.data.uniform_(-stdv_omega, stdv_omega)
        if self.bias is not None:
            raise "Not implimented"
            self.bias.data.uniform_(-stdv, stdv)

    
    def forward(self,x):
        #input = torch.Size([25, 7, 192])
        #input transformed totorch.Size([25, 192, 7])
        #y after CNN = torch.Size([25, 192])
        print('xmod', x.shape)

        theta = self.theta.squeeze(1)
        #print('theta', theta.shape)
        t0 = 1
        t1 = self.laplacian  # torch.Size([192, 192])
        t_sum = 2 * self.laplacian.matmul(t1)  
        t_sum = theta.matmul(t_sum)  # torch.Size([192, 1, 192])
        #print('t_sum pre', t_sum.shape)
        
        for k in range(self.max_diffusion_step):
            t = 2 * self.laplacian.matmul(t1) - t0
            t = theta * t
            t_sum = t_sum + t
            t0, t1 = t1, t
            
        #t_sum = t_sum.unsqueeze(1)
        print('xmod', x.shape)
        print('t_sum', t_sum.shape)
        y = t_sum.matmul(x)
        y = y.unsqueeze(1)
        y = y.matmul(self.omega)
        
        return y
    
    

class CNN_diffusion(BaseNetwork):
    def __init__(self, previous_timesteps, max_diffusion_step):
        super(CNN_diffusion,self).__init__()
        self.road_network = load_network(MoviaBusDataset.hack_filters)
        self.adj_mat = adjacency_matrix(self.road_network)
        #first order approx
        self.n_roads = self.adj_mat.shape[0]
        sparse_mat = calculate_normalized_laplacian(torch.from_numpy(self.adj_mat).float() + torch.eye(self.n_roads))
        self.laplacian = torch.cuda.FloatTensor(sparse_mat.todense()) if torch.cuda.is_available()  else torch.FloatTensor(sparse_mat.todense())#self.convl1 = ConvolutionLayer(self.laplacian,(previous_timesteps+1),1)
        self.relu = ReLU()
        self.CNN = Sequential(
            ConvolutionLayer_diffusion(laplacian=self.laplacian, n_out=self.n_roads, timesteps=previous_timesteps+1, max_diffusion_step=max_diffusion_step),
            ReLU(),
            ConvolutionLayer_diffusion(laplacian=self.laplacian, n_out=self.n_roads, timesteps=1, max_diffusion_step=max_diffusion_step),
            ReLU(),
            #ConvolutionLayer_diffusion(self.laplacian, self.n_roads,1),
            #ReLU(),
            #ConvolutionLayer(self.laplacian, self.n_roads,1),
            #ReLU(),
            #ConvolutionLayer(self.laplacian,num_hidden, 1),
            #Linear(num_hidden*(previous_timesteps+1), self.n_nodes),
            #Linear(num_hidden * (previous_timesteps+1), 1),
        )
        
    
    
    def forward(self,x):
        """
        x : [batch_size, prev_timesteps, num_roads]
        
        """
        #Transpose input, such that the previous time steps are the last dimension
        print('x pre', x.shape)
        x = x.transpose(2,1)
        predictions = []
        for _ in range(self.max_timestep):
            #Run the input through the network
            print('x loop', x.shape)
            y = self.CNN(x).squeeze()

            #Append the prediction to the list of predictions
            predictions.append(y[:,:self.num_roads])

            #remove oldest timestep
            x = x[:,:,1:]


            #unsqueeze output so its size is [batch_size, num_roads, timesteps]
            y = y.unsqueeze(2)
            
            #append the new prediction to the input
            x = torch.cat((x,y),dim=2)
        return torch.stack(predictions,1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# class GraphConvolution(Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """
# 
# 
# 
#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
# 
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
# 
#     def forward(self, input, laplacian):
#         support = input.matmul(self.weight)
#         #support = torch.mm(input, self.weight)
#         output = laplacian.matmul(support)
#         #output = torch.spmm(adj, support)
# 
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output
# 
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'
# 
# class GCN(BaseNetwork):
#     def __init__(self, nfeat, nhid, nout, dropout):
#         super(GCN, self).__init__()
# 
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nout)
#         self.dropout = dropout
#         
#         # Define laplacian to first order approximation
#         self.road_network = load_network(MoviaBusDataset.hack_filters)
#         self.adj_mat = adjacency_matrix(self.road_network)
#         self.n_nodes = self.adj_mat.shape[0]
#         sparse_mat = calculate_normalized_laplacian(self.adj_mat + torch.eye(self.n_nodes))
#         #self.laplacian = torch.tensor(sparse_mat.todense()).cuda()
#         self.laplacian = torch.cuda.FloatTensor(sparse_mat.todense()) if torch.cuda.is_available()  else torch.FloatTensor(sparse_mat.todense())
#     def forward(self, x):
#          #Transpose input, such that the previous time steps are the last dimension
#         x = x.transpose(2,1)
#         
#         predictions = []
#         for _ in range(self.max_timestep):
#             #Run the input through the network
#             tmp = F.relu(self.gc1(x, self.laplacian))
#             tmp = F.dropout(tmp, self.dropout, training=self.training)
#             y = self.gc2(tmp, self.laplacian).squeeze()
#             
#             #Append the prediction to the list of predictions
#             predictions.append(y[:,:self.num_roads])
# 
#             #remove oldest timestep
#             x = x[:,:,1:]
#             
#             #unsqueeze output so its size is [batch_size, num_roads, timesteps]
#             y = y.unsqueeze(2)
# 
#             #append the new prediction to the input
#             x = torch.cat((x, y),dim=2)
# 
# 
#         return torch.stack(predictions,1)

