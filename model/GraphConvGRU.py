from torch.nn.modules.rnn import RNNCellBase
from torch.nn import Module, ModuleList
from torch.nn.functional import sigmoid

class GraphConv(Module):
    def __init__(self, kernels, output_size, max_diffusion_step = 10, bias = True):
        super().__init__()


    def forward(self, input):
        pass

class GraphConvGRUCell(Module):
    ""
    def __init__(self, input_size, hidden_size, kernels,  max_diffusion_step = 10, activation = sigmoid, bias = True):
        
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self._max_diffusion_step = max_diffusion_step

        self.kernels = []

        self.gconv = GraphConv(kernels, hidden_size, max_diffusion_step = max_diffusion_step)
        

    def forward(self, input, hidden):
        """
        Args:
        - input (batch_size, input_size)
        - hidden (batch_size, hidden_size)

        Returns:
        - output (batch_size, hidden)
        - hidden (batch_size, hidden)
        """

        
        #We want to do graph convolution on both input and hidden state, so concat to one matrix
        combined = torch.cat([input, hidden], dim=1)
        
        #Do two separate graph convolutions to get r and u
        r = sigmoid(self.gconv(combined))
        u = sigmoid(self.gconv(combined))

        #to calculate c we need [X, r*H]
        combined2 = torch.cat([input, r*hidden])

        #do a final graph convolution
        C = self.activation(self.gconv(combined2))

        #the output and hidden state of the GRU can now be updated
        output = hidden = u * hidden + (1 - u) * C

        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.batch_size, self.hidden_size)




class GraphConvGRU(Module):

    def __init__(self, input_size, hidden_size, kernels, num_layers=1, bias=True, batch_first = False, dropout = 0):
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

            cells.append(GraphConvGRUCell(input_size=current_input_size,
                                          hidden_size=hidden_size, 
                                          kernels=kernels, 
                                          bias=bias))

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
        
        batch_size, seq_len, _ = input.size()

        #create tensor for output and final hidden state
        output = torch.zeros(seq_len, batch_size, self.hidden_size)
        hidden_out = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        if self.batch_first:
            #move batch to first dim
            input = input.permute(1,0,2)
            output = output.permute(1,0,2)
            hidden_out = hidden_out.permute(1,0,2)
            hidden = hidden.permute(1,0,2)

        if hidden is None:
            hidden = self.init_hidden(batch_size)

        

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

        return output, hidden_out


    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.num_layers, self.hidden_size)