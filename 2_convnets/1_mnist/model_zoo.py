import torch.nn as nn
    
class small_cnn(nn.Module):
    # Model
    def __init__(self):
        super(small_cnn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=1),
            nn.ReLU(True),  
            ) 

        self.classifier = nn.Sequential(
            nn.Linear(10*28*28, 100),
            nn.ReLU(True),
            nn.Linear(100, 10),
            nn.Softmax()
            )
        
    def forward(self, x):
        x = x.view([x.size(0),1,28,28])
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        
class medium_cnn(nn.Module):
    # Model
    def __init__(self):
        super(small_cnn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=1),
            nn.ReLU(True),  
            nn.Conv2d(10, 20, kernel_size=3, padding=1),
            nn.ReLU(True)  
            ) 

        self.classifier = nn.Sequential(
            nn.Linear(20*28*28, 100),
            nn.ReLU(True),
            nn.Linear(100, 10),
            nn.Softmax()
            )
        
    def forward(self, x):
        x = x.view([x.size(0),1,28,28])
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x