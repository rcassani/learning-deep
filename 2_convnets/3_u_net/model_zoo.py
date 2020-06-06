import torch.nn as nn
import torch

class PrintLayerSize(nn.Module):
    def __init__(self):
        super(PrintLayerSize, self).__init__()
        
    def forward(self, x):
#        print(x.shape)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.seq = nn.Sequential(
            PrintLayerSize(),
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=(4,2)),
            nn.ReLU(True)
            )
        
    def forward(self, x):
          return self.seq(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.seq = nn.Sequential(
            PrintLayerSize(),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=2, padding=(4,2)),
            nn.ReLU(True)
            )
        
    def forward(self, x, x2):
          return self.seq(torch.cat([x2, x], dim=1))


    
class unet_small(nn.Module):
    # Model
    def __init__(self):
        super(unet_small, self).__init__()
               
        self.down1 = Down(1, 10) 
        self.down2 = Down(10, 20) 
        self.down3 = Down(20, 30)
        
        self.up3 = nn.Sequential(
            PrintLayerSize(),
            nn.ConvTranspose2d(30, 20, kernel_size=5, stride=2, padding=(4,2)),
            nn.ReLU(True)  
            )            
        
        self.up2 = Up(40, 10)        
        self.up1 = Up(20, 1)
                    
        self.out = nn.Sequential(
            PrintLayerSize(),
            nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1),
            nn.Softmax(),
            PrintLayerSize()
            )      


    def forward(self, x):        
        # down
        x1 = self.down1(x)               
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        # up
        x = self.up3(x3)
        x = self.up2(x, x2)
        x = self.up1(x,x1)
        # out
        x = self.out(x)
        return x
