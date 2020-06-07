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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            PrintLayerSize(),
            )
        
    def forward(self, x):
          return self.seq(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.seq = nn.Sequential(
            PrintLayerSize(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            PrintLayerSize(),
            )
        
    def forward(self, x, x2):
          x = nn.functional.upsample(x, scale_factor=2, mode='bilinear',align_corners=True)
          return self.seq(torch.cat([x2, x], dim=1))


    
class unet_small(nn.Module):
    # Model
    def __init__(self):
        super(unet_small, self).__init__()
               
        self.down1 = Down(1, 64) 
        self.down2 = Down(64, 128) 
        self.down3 = Down(128, 256)
        
        self.up3 = nn.Sequential(
            PrintLayerSize(),          
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            )            
        
        self.up2 = Up(256, 64)        
        self.up1 = Up(128, 1)
                    
        self.out = nn.Sequential(
            PrintLayerSize(),
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1),
            nn.Softmax(dim=1),
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
        x = self.up1(x, x1)
        # out
        x = self.out(x)
        return x
