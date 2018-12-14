import torch
from torch.nn import Linear, Sequential, ReLU, L1Loss, BatchNorm1d, Dropout2d, GRU
from BaseNetwork import BaseNetwork
from torch.optim import Adam
import random


class Seq2Seq(BaseNetwork):    
    def __init__(self, hidden_size=100, num_layers=1,num_roads=192,prev_timesteps=6,prediction_steps=6):
        super().__init__(name="Sequence2Sequence")
        
        self.prev_timesteps = prev_timesteps
        self.num_roads = num_roads
        self.prediction_steps=prediction_steps
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.encoder = GRU(num_roads, hidden_size, batch_first=True, num_layers=num_layers)
        
        self.decoder = GRU(num_roads, hidden_size, batch_first=True, num_layers=num_layers)
        #self.activation = Sig()
        self.decoder_l1 = Linear(hidden_size, num_roads)
        
        self.criterion = L1Loss()
        
    def forward(self, x_in):
        x = x_in['data']
        target = x_in['target']
        
        n_batch = x.size()[0]
        #print(x.size())
        #x = self.BN(x)
        #torch.manual_seed(42)
        #hidden = (torch.zeros(self.num_layers, n_batch, self.hidden_size),#.cuda(),
        #          torch.zeros(self.num_layers, n_batch, self.hidden_size))#.cuda())
        hidden = torch.zeros(self.num_layers, n_batch, self.hidden_size)
        #Run previous timesteps through the encoder
        for t_i in range(self.prev_timesteps):
            _, hidden = self.encoder(x[:,t_i,:].view(-1,1,self.num_roads),hidden)
            
        
        #Use a GO symbol for the first input to the decoder
        x_t = torch.zeros(n_batch, 1, self.num_roads)
        
        use_teacher_forcing = True if random.random() < 0.5 and self.training else False
        predictions = []
        #Use the model to predict several timesteps into the future
        for t in range(self.prediction_steps):            
            #run through LSTM
            
            x_out, hidden = self.decoder(x_t.view(-1,1,self.num_roads),hidden)
            
            #apply activation and final outout layer
            x_out = self.decoder_l1((x_out))
                        
            prediction = x_out[:,0,:self.num_roads]
        
            predictions.append(prediction)
      
            #Use teacher forcing where we use the target at t to predict t+1
            if use_teacher_forcing:
                x_t = target[:,t,:]
            #Otherwise we use the prediction as the input 
            else:
                x_t = prediction
    
        return torch.stack(predictions,1)