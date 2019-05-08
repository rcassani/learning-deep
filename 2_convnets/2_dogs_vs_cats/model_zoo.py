import torch.nn as nn
    
class small_cnn(nn.Module):
    # Model
    def __init__(self):
        super(small_cnn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, padding=0, stride=3),
            nn.ReLU(True),  
            ) 

        self.classifier = nn.Sequential(
            nn.Linear(10*21*21, 100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.Softmax()
            )
        
    def forward(self, x):
        #x = x.view([x.size(0),1,28,28])
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
                

class medium_cnn(nn.Module):
    # Model
    def __init__(self):
        super(medium_cnn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),  
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 20, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            ) 

        self.classifier = nn.Sequential(
            nn.Linear(20*16*16, 100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.Softmax()
            )
        
    def forward(self, x):
        #x = x.view([x.size(0),1,28,28])
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class large_cnn(nn.Module):
    # Model
    def __init__(self):
        super(large_cnn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(True),  
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            ) 

        self.classifier = nn.Sequential(
            nn.Linear(512*2*2, 100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.Softmax()
            )
        
    def forward(self, x):
        #x = x.view([x.size(0),1,28,28])
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    