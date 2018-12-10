from pytorch.nn import Module

class DCGruCell(Module):
    def __init__(self,max_diffusion_step=10):
        super().__init__()
        self._max_diffusion_step = max_diffusion_step

    def _graph_conv(self, input, kernel):
        """
        
        input [batch_size, num_roads]
        kernel [num_roads]
        """


