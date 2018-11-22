import torch

#from torch.autograd import Variable
#from torch.nn.parameter import Parameter
from BaseNetwork import BaseNetwork
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from torch.nn.modules import Module

#from torch.nn import Linear, Conv1d, BatchNorm1d, MaxPool1d, Dropout
#from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.nn import Module, Linear, ReLU, Sequential
from BaseNetwork import BaseNetwork
from torch.optim import Adam
import sys
sys.path.append("../DCRNN")
from lib.utils import calculate_normalized_laplacian
from misc.data_loader import load_network, adjacency_matrix


class ConvolutionLayer(Module):
    def __init__(self,laplacian,n_signals,n_out):
        super(ConvolutionLayer,self).__init__()
        self.n_nodes = laplacian.shape[0]
        self.laplacian = laplacian
        self.l1  = Linear(n_signals * self.n_nodes, n_out * self.n_nodes,bias = False)
    
    def forward(self,x):
        y = self.laplacian.matmul(x)
        shape = y.shape
        y = y.reshape(shape[0],shape[1]*shape[2])
        y = self.l1(y)
        y = y.reshape(shape)
        return y


class CNN(BaseNetwork):
    def __init__(self, previous_timesteps, num_hidden):
    #def __init__(self, num_hidden):
        super(CNN,self).__init__()
        self.road_network = load_network()
        self.adj_mat = adjacency_matrix(self.road_network)
        #first order approx
        self.n_nodes = self.adj_mat.shape[0]
        sparse_mat = calculate_normalized_laplacian(self.adj_mat + torch.eye(self.n_nodes))
        self.laplacian = torch.tensor(sparse_mat.todense()).float()
        #self.convl1 = ConvolutionLayer(self.laplacian,(previous_timesteps+1),1)
        self.relu = ReLU()
        self.CNN = Sequential(
            ConvolutionLayer(self.laplacian,previous_timesteps+1, previous_timesteps+1),
            ReLU(),
            ConvolutionLayer(self.laplacian, previous_timesteps+1,previous_timesteps+1),
            ReLU(),
            Linear((previous_timesteps+1), 1),
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
            y = y.unsqueeze(2)

            #append the new prediction to the input
            x = torch.cat((x,y),dim=2)


        return torch.stack(predictions,1)


