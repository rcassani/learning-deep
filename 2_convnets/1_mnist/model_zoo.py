import torch.nn as nn
import torch.nn.functional as F
    
class small_cnn(nn.Module):
    # Model
    def __init__(self):
        super(small_cnn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=1),
            nn.AlphaDropout(p=0.5),
            nn.SELU() 
            ) 

        self.fc1 = nn.Linear(10*28*28, 100)
        self.fc2 = nn.Linear(100, 10)
        
    def forward(self, x):
        x = x.view([x.size(0),1,28,28])
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
        
