import torch.nn as nn
import torch.nn.functional as F

class simple_mlp(nn.Module):
    # Model
    def __init__(self):
        super (simple_mlp, self).__init__()
        # Layers					
        self.fc1 = nn.Linear(28*28 , 100)
        self.fc2 = nn.Linear(100 , 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x 

class deep_mlp(nn.Module):
    # Model
    def __init__(self):
        super (deep_mlp, self).__init__()
        # Layers					
        self.fc1 = nn.Linear(28*28 , 520)
        self.fc2 = nn.Linear(520 , 320)
        self.fc3 = nn.Linear(320 , 240)
        self.fc4 = nn.Linear(240 ,  10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x
    
class small_cnn_mnist(nn.Module):
    # Model
    def __init__(self):
        super(small_cnn_mnist, self).__init__()
        self.conv_1 = nn.Conv2d(1, 6, kernel_size=3, padding=0)
        self.pool_1 = nn.MaxPool2d(2, 2)
        self.conv_2 = nn.Conv2d(6, 16,kernel_size=5, padding=0)
        self.fcnx_1 = nn.Linear(16 * 4 * 4, 120)
        self.fcnx_2 = nn.Linear(120, 84)
        self.fcnx_3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x.view([x.size(0),1,28,28])
        x = F.relu(self.conv_1(x))
        x = self.pool_1(x)
        x = F.relu(self.conv_2(x))
        x = self.pool_1(x)         
        x = x.view(x.size(0), -1) #flats
        x = F.relu(self.fcnx_1(x))
        x = F.relu(self.fcnx_2(x))
        x = self.fcnx_3(x)
        return x
        
