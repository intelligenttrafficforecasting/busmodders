from torch.nn import Module
from torch.nn.functional import sigmoid

class GraphConvGRUCell(Module):
    ""
    def __init__(self, max_diffusion_step = 10, activation = sigmoid):
        super().__init__()
        self._max_diffusion_step = max_diffusion_step


    def forward(self, x, h=None):
        """"""
        if h is None:
           #h = x.new_zeros(x.size(0))

        combined = torch.cat([x, h], dim=1)

    def _graph_conv(self, input, kernel):
        """
        
        input [batch_size, num_roads]
        kernel [num_roads]
        """


class GraphConvGRU(Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first = False, dropout = 0):