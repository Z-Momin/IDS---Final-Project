import torch
import torchvision.models as models
from torch import nn


def conv_layer(c_in, c_out, ks=3, stride=2):
    return nn.Sequential(
                        nn.Conv2d(c_in, c_out, ks, padding=ks//2, stride=stride, bias=False),
                        nn.ReLU(),
                        nn.BatchNorm2d(c_out, eps=1e-5, momentum=0.1)
                        )
                        

class Flatten(nn.Module):

    def __init(self):
        super().__init__()
    
    def forward(self, x): return x.view(x.size(0), -1)


class Model(nn.Module):
    
    def __init__(self, c_layers, nh=50, c_in=4, c_out=9, p=0.5, imsize=(96, 96)):
        
        super().__init__()
        c_layers = [c_in] + c_layers
        self.cnn = nn.Sequential(*[conv_layer(c_layers[i], c_layers[i+1], ks=5 if i==0 else 3, stride=1 if i==0 else 2) for i in range(len(c_layers)-1)], nn.Flatten())
        

        with torch.no_grad():
            xb_tmp = torch.randn(1, c_in, *imsize)
            y_pred = self.cnn(xb_tmp)
            ni = y_pred.size(-1)
        self.fcn = nn.Sequential(nn.Linear(ni, nh), nn.ReLU(), nn.Dropout(p=p), nn.Linear(nh, c_out))
        
    def forward(self, x): 
        x = self.cnn(x)
        return self.fcn(x)


class Model_v2(nn.Module):
    
    def __init__(self, c_layers, nh=50, c_in=4, c_out=3, p=0.5, imsize=(96, 96)):
        
        super().__init__()
        c_layers = [c_in] + c_layers
        self.cnn = nn.Sequential(*[conv_layer(c_layers[i], c_layers[i+1], ks=5 if i==0 else 3, stride=1 if i==0 else 2) for i in range(len(c_layers)-1)], nn.Flatten())
        

        with torch.no_grad():
            xb_tmp = torch.randn(1, c_in, *imsize)
            y_pred = self.cnn(xb_tmp)
            ni = y_pred.size(-1)
        self.steer = nn.Sequential(nn.Linear(ni, nh), nn.ReLU(), nn.Dropout(p=p), nn.Linear(nh, c_out))
        self.accelerate = nn.Sequential(nn.Linear(ni, nh), nn.ReLU(), nn.Dropout(p=p), nn.Linear(nh, c_out))
        
    def forward(self, x): 
        x = self.cnn(x)
        return self.steer(x), self.accelerate(x)
