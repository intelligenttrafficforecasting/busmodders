from torch.nn import Module, L1Loss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import torch
import numpy as np


def has_cuda():
    """Hack to check if CUDA is available. This fallbacks to CPU if you have GPU, but it's too old"""
    import warnings
    try:
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings('error')
            torch.cuda._check_capability()
            return True
    except:
        return False

        
class BaseNetwork(Module):
    """
    Base class for out networks. This contains shared methods like
        - train_network: Method used for training the network
        - get_MAE_score: Returns the MeanAbsoluteError on the test data
    """
    def __init__(self):
        super().__init__()

        #initialize class variables for readibility
        self.max_timestep = 0
        self.num_roads = 0
       
        if has_cuda():
           self.cuda()

    def train_network(self,\
            train, 
            validation,
            batch_size = 50, 
            num_epochs = 100, 
            optimizer_fun = lambda param: Adam(param), 
            scheduler_fun = None, 
            criterion = L1Loss(),
            shuffle = False,
            target_to_net = False
            ):

        """
        Base method for training any network which inherits from this class

        Args:
            train (dataset): Training dataset. Should be a torch dataset
            test (dataset): validation dataset. Should be a torch dataset
            batch_size (int): The batch_size used for training.
            num_epochs (int): Number of epochs to run the training for
            optimizer_fun (function): The function used to contruct the optimizer. 
                                    Should take the parameters of the network as input
            criterion (function): The loss function that should be used
            shuffle (Boolean): Flag to shuffle the data or not
            target_to_net (Boolean): Give the targets as input to the forward function in the network. Used for curriculum learning
        
        """    

        #Save some of the informations in the class for later
        self.validation_data = validation
        self.train_data = train
        self.criterion = criterion
        self.__target_to_net = target_to_net

        #initialize the optimizer
        self.optimizer = optimizer_fun(self.parameters())

        if scheduler_fun is not None:
            self.scheduler = scheduler_fun(self.optimizer)
        else:
            self.scheduler = None

        #Get how many time steps in the future we want to predict, and the number of roads
        self.max_timestep, self.num_roads = train[0]['target'].size()
        
        #Create the data loaders
        train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
        validation_dataloader = DataLoader(validation, batch_size=batch_size, shuffle=shuffle)

        #Enable CUDA
        if has_cuda():
           self.cuda()

        
        for epoch in range(num_epochs):
            self.train()
            train_loss = []
            validation_loss = []
            #Train on the training dataset
            for _ , batch_train in enumerate(train_dataloader):
                #Get the predictions and targets
                if self.__target_to_net:
                    output = self(batch_train)
                else:
                    output = self(batch_train['data'])
                target = batch_train['target']

                #set the gradients to zero
                self.optimizer.zero_grad()

                #Calculate the loss, and backpropagate
                loss = criterion(output, target)   
                loss.backward()
                
                #Optimize the network
                self.optimizer.step()

                
                train_loss.append(loss.item())

            #Evaluate the results on the validation set
            self.eval()
            for _, batch_validation in enumerate(validation_dataloader):
                if self.__target_to_net:
                    output = self(batch_validation)
                else:
                    output = self(batch_validation['data'])

                target = batch_validation['target']
                loss = criterion(output, target)   
                validation_loss.append(loss.item())

            if self.scheduler is not None:
                    self.scheduler.step()

            if epoch % 2 == 0:
                print("epoch = %2i  train loss = %0.3f   validation loss = %0.3f   output_std = %0.3f" %(epoch, np.mean(train_loss), np.mean(validation_loss) , output.std().item()))

            

    def get_MAE_score(self, dataset, timestep = 1,individual_roads=False):
        """
        Returns the MeanAbsoluteError on the test dataset

        Args:
            timestep (int): How many timesteps into the future you want to calculate the error for
        """
        device = torch.device('cpu')
        if has_cuda():
           device = torch.device('cuda')
        
        DL = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        self.eval()
        for _, batch in enumerate(DL):
            if self.__target_to_net:
                output = self(batch)
            else:
                output = self(batch['data'])
            target = batch['target']
                
            #If input is normalized, we need to denormalize it
            if dataset.std is not None:
                output = output*torch.tensor(dataset.std, device=device) + torch.tensor(dataset.mean, device=device)
                target = target*torch.tensor(dataset.std, device=device) + torch.tensor(dataset.mean, device=device)
            if individual_roads:
                o = output[:,timestep-1,:]
                t = target[:,timestep-1,:]
                n_roads = o.shape[1]
                loss = np.empty(n_roads)
                for i in range(n_roads):
                    loss[i] = self.criterion(output[:,timestep-1,i],target[:,timestep-1,i]).item()
                return loss
            else:
                loss = self.criterion(output[:,timestep-1,:],target[:,timestep-1,:])
                return loss.item()

    def visualize_road(self, dataset, timesteps=1, road=1):
        """
        Visualizes the predictions compared to the ground truth for a particular road

        Args:
            timesteps (int): How many timesteps into the future you are comparing
            road (int): index of the road you want to show
        """

        import matplotlib.pyplot as plt
        from datetime import datetime

        DL = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        device = torch.device('cpu')
        if has_cuda():
           device = torch.device('cuda')
        
        self.eval()
        for _, batch in enumerate(DL):
            if self.__target_to_net:
                output = self(batch)
            else:
                output = self(batch['data'])
            target = batch['target']
            time = batch['time']
            #If input is normalized, we need to denormalize it
            if dataset.std is not None:
                output = output*torch.tensor(dataset.std, device=device) + torch.tensor(dataset.mean, device=device)
                target = target*torch.tensor(dataset.std, device=device) + torch.tensor(dataset.mean, device=device)
        #time is in seconds, so create a function that can convert it into datetimes again
        seconds_to_datetime = np.vectorize(datetime.fromtimestamp)
        time = seconds_to_datetime(time)
        
        plt.plot(time[:,timesteps-1],output[:,timesteps-1,road].detach().cpu().numpy(), label='Prediction')
        plt.plot(time[:,timesteps-1],target[:,timesteps-1,road].detach().cpu().numpy(), label='Truth')
        plt.legend()
        plt.show()