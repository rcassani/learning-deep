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
    
class small_cnn(nn.Module):
    # Model
    def __init__(self):
        super(small_cnn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=1),
            nn.AlphaDropout(p=0.5),
            nn.SELU() ) 

        self.fc1 = nn.Linear(10*28*28, 100)
        self.fc2 = nn.Linear(100, 10)
        
    def forward(self, x):
        x = x.view([x.size(0),1,28,28])
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
        
