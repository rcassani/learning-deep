import torch.nn as nn
import torch.nn.functional as F

class mlp_notebook(nn.Module):
    # Model
    def __init__(self):
        super (mlp_notebook, self).__init__()
        # Layers					
        self.fc1 = nn.Linear(28*28 , 25)
        self.fc2 = nn.Linear(25 , 10)
        self.fc3 = nn.Linear(10 , 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) 
        
        return x 

        
