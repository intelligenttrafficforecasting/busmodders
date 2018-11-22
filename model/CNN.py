

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

class CNN(BusmodderNet):

    def __init__(self):
        super(CNN, self).__init__()
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
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

            #append the new prediction to the input
            x = torch.cat((x,y),dim=2)


        return torch.stack(predictions,1)



    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, laplacian):
        support = input.matmul(self.weight)
        #support = torch.mm(input, self.weight)
        output = laplacian.matmul(support)
        #output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(BaseNetwork):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout
        
        # Define laplacian to first order approximation
        self.road_network = load_network()
        self.adj_mat = adjacency_matrix(self.road_network)
        self.n_nodes = self.adj_mat.shape[0]
        sparse_mat = calculate_normalized_laplacian(self.adj_mat + torch.eye(self.n_nodes))
        self.laplacian = torch.tensor(sparse_mat.todense()).float()
        
    def forward(self, x):
         #Transpose input, such that the previous time steps are the last dimension
        x = x.transpose(2,1)
        
        predictions = []
        for _ in range(self.max_timestep):
            #Run the input through the network
            tmp = F.relu(self.gc1(x, self.laplacian))
            tmp = F.dropout(tmp, self.dropout, training=self.training)
            y = self.gc2(tmp, self.laplacian).squeeze()
            
            #Append the prediction to the list of predictions
            predictions.append(y[:,:self.num_roads])

            #remove oldest timestep
            x = x[:,:,1:]
            
            #unsqueeze output so its size is [batch_size, num_roads, timesteps]
            y = y.unsqueeze(2)

            #append the new prediction to the input
            x = torch.cat((x, y),dim=2)


        return torch.stack(predictions,1)

