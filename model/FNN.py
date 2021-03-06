import torch
from torch.nn import Linear, Sequential, ReLU, L1Loss, BatchNorm1d, Dropout2d
from BaseNetwork import BaseNetwork
from torch.optim import Adam

class FNN(BaseNetwork):
    def __init__(self, num_hidden, previous_timesteps = 6):
        super().__init__()
        
        self.FNN = Sequential(
            #
            Linear(previous_timesteps+1, num_hidden),
            ReLU(),
            #Linear(num_hidden, 2*num_hidden),
            #ReLU(),
            #BatchNorm1d(193),
            #Dropout2d(0.1),
            #Linear(2*num_hidden, num_hidden),
            Linear(num_hidden, num_hidden),
            ReLU(),
            #Dropout2d(0.1),
            #BatchNorm1d(193),
            Linear(num_hidden, 1),
            )
        
        self.criterion = L1Loss()
           
    def forward(self,x):
        """
        x : [batch_size, prev_timesteps, num_roads]
        
        """
        #Transpose input, such that the previous time steps are the last dimension
        x = x.transpose(2,1)

        predictions = []
        for _ in range(self.max_timestep):
            #Run the input through the network
            prediction = self.FNN(x).squeeze()

            #Append the prediction to the list of predictions. 
            #If the data includes timeofday, this shouldn't be included
            predictions.append(prediction[:,:self.num_roads])

            #remove oldest timestep
            x = x[:,:,1:]
            #unsqueeze output so its size is [batch_size, num_roads, timesteps]
            prediction = prediction.unsqueeze(2)

            #append the new prediction to the input
            x = torch.cat((x,prediction),dim=2)

        return torch.stack(predictions,1)


if __name__ == '__main__':
    #Example usage
    import sys
    sys.path.append('misc')
    from MoviaBusDataset import MoviaBusDataset


    prev_timesteps = 6
    prediction_steps = 6

    train = MoviaBusDataset('data/train', interpolation=True, 
                            prev_timesteps=prev_timesteps, 
                            max_future_time_steps=prediction_steps, 
                            timeofday = True)

    test = MoviaBusDataset('data/test', interpolation=True, 
                            prev_timesteps=prev_timesteps, 
                            max_future_time_steps=prediction_steps, 
                            timeofday = True)

    train.normalize()
    test.normalize(train.mean, train.std)

    fnn = FNN(num_hidden=100, previous_timesteps=prev_timesteps)
    fnn.train_network(train, test, optimizer_fun=lambda param: Adam(param, lr=1e-2), num_epochs=10)
    fnn.get_MAE_score()
    fnn.visualize_road(1,1)