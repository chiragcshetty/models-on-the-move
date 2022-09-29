import torch
from torch import optim, nn

class basicModel(nn.Module):
    def __init__(self, factor) -> None:
        super().__init__() # python 3 syntax
        
        self.factor = factor
        self.linear1N = int(64*self.factor)
        self.linear2N = int(128*self.factor)
        self.linear3N = int(24*self.factor)
        self.linear4N = int(10*self.factor)

        self.fc1 = nn.Linear(self.linear1N, self.linear1N)
        self.fc2 = nn.Linear(self.linear1N, self.linear2N)
        self.fc3 = nn.Linear(self.linear2N, self.linear3N)
        self.fc4 = nn.Linear(self.linear3N, self.linear4N)

    def forward(self, x):
        x = x.squeeze()
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

def get_model(model_name, *args, **kwargs):
    if model_name == "basicModel":
        return basicModel(*args, **kwargs)

def get_optimizer(optimizer_name, *args, **kwargs):
    if optimizer_name == "SGD":
        return optim.SGD(*args, **kwargs)

def get_criterion(criterion_name, *args, **kwargs):
    if criterion_name == "MSELoss":
        return nn.MSELoss(*args, **kwargs)