from torch.nn.modules.rnn import RNNCellBase
from torch.nn import Module, ModuleList, Parameter, Linear
from torch import sigmoid, tanh
import torch

class GraphDiffusionConv(Module):
    def __init__(self, input_size, kernels, output_size = 1, max_diffusion_step = 2, bias = True):
        """
        Args:
        - input_size (int) : the size of the input graph (number of nodes)
        - kernels (list of tensors): list of the kernels that should be used in the convolution
        - output_size (int): Output size of the convolution. This corresponds to a number for each node in the graph, so currently this is fixed to 1
        - max_diffusion_step (int): Number of diffusion steps to do in the convolution
        - bias (boolean): Flag if the layer should use a bias
        
        """
        super().__init__()
 
        if output_size != 1:
            raise NotImplementedError("GraphDiffusionConv currenly only support one output from the graph")
        
        self.input_size = input_size
        self.max_diffusion_step = max_diffusion_step
        self.kernels = kernels
        self.k_tot = len(self.kernels) * self.max_diffusion_step + 1

        self.linear = Linear(in_features = 2 * self.k_tot, out_features = output_size, bias = bias) #the 2 is since we use both the input and hidden state


    def forward(self,input, hidden = None):
        """
        Args:
        - input (batch_size, input_size)
        - hidden (batch_size, output_size) : When used with recurrent networks, the hidden state can be convolved at the same time
        Returns:
        - output (batch_size, input_size)
        """

        if hidden is  None:
            raise NotImplementedError("GraphDiffusionConv only support RNN use, where both the input and hidden state is given")       

        batch_size = input.size(0)

        #stack input and hidden state to (batch_size, input_size, 2)
        input_state = torch.stack([input, hidden], dim=2)
        num_inputs = input_state.size(2)

        x = input_state
        x0 = x#.permute(1,2,0) # -> (input_size, 2, batch_size)
        x = x0.unsqueeze(0) # -> (k_tot, input_size, 2, batch_size)
        for kernel in self.kernels:
            x1 = kernel.matmul(x0)
            x = torch.cat((x, x1.unsqueeze(0)))

            for k in range(2, self.max_diffusion_step + 1):
                x2 = 2 * kernel.matmul(x1) - x0
                x = torch.cat((x, x2.unsqueeze(0)))
                x1, x0 = x2, x1


        #the total number of diffusion steps across all kernels
        k_tot = len(self.kernels) * self.max_diffusion_step + 1

        x = x.view(k_tot, self.input_size, num_inputs, batch_size)
        x = x.permute(3,1,2,0) # -> (batch_size, input_size, num_input, k_tot)
        #reshape to 2d matrix, where each row corresponds to the one observation of the graph
        x = x.contiguous().view(batch_size*self.input_size, num_inputs * k_tot)

        #apply matmul + bias to get (batch_size*input_size, 1)
        x = self.linear(x)

        #finaly convert back to (batch_size, input_size)
        x = x.view(batch_size, self.input_size)
        return x

class GraphConvGRUCell(Module):
    ""
    def __init__(self, input_size, hidden_size, kernels,  max_diffusion_step = 2, activation = tanh, bias = True):
        """
        
        Args:
        - input_size (int): input size of the graph (number of nodes)
        - hidden_size (int): The size of the output from the cell. Currently the output is fixed to the same as input
        - kernels (list of tensors): The kernels that should be used for the convolution
        - max_diffusion_step (int): number of diffusion steps to do in the convolution
        - activation (function): The activation function to apply to the C gate in the cell. defaults to tanh
        - bias (boolean): Flag if the layer should use a bias
        
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self._max_diffusion_step = max_diffusion_step

        #we need to do 3 graph convolutions
        self.gconv1 = GraphDiffusionConv(input_size, kernels = kernels, max_diffusion_step = max_diffusion_step, bias = bias)
        self.gconv2 = GraphDiffusionConv(input_size, kernels = kernels, max_diffusion_step = max_diffusion_step, bias = bias)
        self.gconv3 = GraphDiffusionConv(input_size, kernels = kernels, max_diffusion_step = max_diffusion_step, bias = bias)
        
        self.linear = Linear(in_features = input_size, out_features = hidden_size)
    def forward(self, input, hidden):
        """
        Args:
        - input (batch_size, input_size)
        - hidden (batch_size, input_size)

        Returns:
        - output (batch_size, hidden_size)
        - hidden (batch_size, input_size)
        """

        #print(input.size())
        #print(hidden.size())

        #Do two separate graph convolutions to get r and u
        r = sigmoid(self.gconv1(input, hidden))
        u = sigmoid(self.gconv2(input, hidden))


        #do a final graph convolution
        C = self.activation(self.gconv2(input, r * hidden))

        #the output and hidden state of the GRU can now be updated
        output = hidden = u * hidden + (1 - u) * C

        #in case we want a different output size than intput
        if self.hidden_size != self.input_size:
            output = self.linear(output)

        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.batch_size, self.hidden_size)




class GraphConvGRU(Module):

    def __init__(self, input_size, hidden_size, kernels, max_diffusion_step=2, num_layers=1, bias=True, batch_first = False, dropout = 0):
        """
        
        Args:
        - input_size (int): input size of the graph (number of nodes)
        - hidden_size (int): The size of the output from the cell. Currently the output is fixed to the same as input
        - kernels (list of tensors): The kernels that should be used for the convolution
        - max_diffusion_step (int): number of diffusion steps to do in the convolution
        - num_layers (int): Number of GRU cells to stack on top of each others
        - bias (boolean): Flag if the layer should use a bias
        - batch_first (boolean): Flag if the input has batch as the first dimension
        - dropout (float): the dropout percentage to apply between the layers. Currently doesn't work
        
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout

        #To support both single kernel, and a list of kernels, make sure the input is a list
        if not isinstance(kernels, list):
            kernels = [kernels]

        cells = []

        #create the cells for the number of layers
        for i in range(num_layers):
            #first layer should have input_size as input, while all others have hidden_size 
            current_input_size = input_size if i==0 else hidden_size

            cells.append(GraphConvGRUCell(input_size = current_input_size,
                                          hidden_size = hidden_size, 
                                          max_diffusion_step = max_diffusion_step,
                                          kernels = kernels, 
                                          bias = bias))

        self.cells = ModuleList(cells)
    
    def forward(self, input, hidden = None):
        """
        Args:
        - input (seq_len, batch_size , input_size)
        - hidden (num_layers, batch_size, hidden_size)

        Returns:
        - output (seq_len, batch_size, hidden_size)
        - hidden_out (num_layers, batch_size, hidden_size)
        """
        

        if self.batch_first:
            #move batch to second dim
            input = input.permute(1,0,2)

            if hidden is not None:
                hidden = hidden.permute(1,0,2)

        seq_len, batch_size, _ = input.size()

        if hidden is None:
            hidden = self.init_hidden(batch_size)

        

        #create tensor for output and final hidden state
        output = torch.zeros(seq_len, batch_size, self.hidden_size)
        hidden_out = torch.zeros(self.num_layers, batch_size, self.hidden_size)


        #print(hidden.size())

        #loop over each layer
        for layer in range(self.num_layers):
            output_layer = []
            #set the initial hidden state for the layer
            h = hidden[layer,:,:]

            #run each value of the sequence through the layer
            for t in range(seq_len):
                out, h = self.cells[layer](input = input[t,:,:], hidden = h)

                #in order to get the output sequence of the final layer, store the output for current layer, and delete them in the begining of the next
                output_layer.append(out)
        
            #h is (batch_size, hidden_size)
            hidden_out[layer,:,:] = h

        #convert list of (batch_size, hidden) into (seq_len, batch_size, hidden)
        output = torch.stack(output_layer, dim=0)

        if self.batch_first:
            #move batch to first dim
            output = output.permute(1,0,2)
            hidden_out = hidden_out.permute(1,0,2)

        return output, hidden_out


    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

if __name__ == "__main__":
    import sys
    sys.path.append("DCRNN")
    sys.path.append("misc")
    from MoviaBusDataset import MoviaBusDataset
    from lib.utils import calculate_normalized_laplacian
    from data_loader import load_network, adjacency_matrix

    a = torch.rand(25, 6, 192)
    road_network = load_network(MoviaBusDataset.hack_filters, path='data/road_network.geojson')
    adj_mat = adjacency_matrix(road_network)

    sparse_mat = calculate_normalized_laplacian(torch.tensor(adj_mat, dtype=torch.float) + torch.eye(192))
    laplacian = torch.tensor(sparse_mat.todense(), dtype=torch.float)
    rnn = GraphConvGRU(input_size=192, hidden_size=192, kernels=[laplacian], batch_first=True)
    rnn(a)