import os
import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as T

######################################################################
# Commonly Used Functions                                            #
######################################################################
def getFlat(image_tensor:torch.Tensor):
    """
    Args:
        image_tensor (torch.Tensor):
    Returns:
        flattened_tensor (torch.Tensor)
    """
    assert type(image_tensor) is torch.Tensor,\
        'image_tensor type must be torch.Tensor'
    assert len(image_tensor.shape) == 3 or len(image_tensor.shape) == 4,\
        'image_tensor_shape = (B, C, H, W) or (C, H, W)'
    
    if len(image_tensor.shape) == 4:
        # If batched image tensor
        return image_tensor.view(image_tensor.shape[0], -1)
    
    elif len(image_tensor.shape) == 3:
        # Single batch image
        return image_tensor.view(-1)


######################################################################
# AutoEncoder with MLP                                               #
######################################################################
class mlpEncoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(mlpEncoder, self).__init__()
        
    def forward(self, x):
        return
    
class mlpDecoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(mlpDecoder, self).__init__()
        
    def forward(self, x):
        return
    
class mlpAutoEncoder(nn.Module):
    def __init__(self, imgsz:tuple):
        super(mlpAutoEncoder, self).__init__()
        self.imgsz= imgsz
        self.encoder = None
        self.decoder = None
        self.fc = nn.Sequential([
            
        ])
        
    def forward(self, x):
        return


######################################################################
# AutoEncoder with Convolutional Layers                              #
######################################################################
class convEncoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(convEncoder, self).__init__()
        
    def forward(self, x):
        return

class convDecoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(convDecoder, self).__init__()
        
    def forward(self, x):
        return
    
class convAutoEncoder(nn.Module):
    def __init__(self, imgsz:tuple):
        super(convEncoder, self).__init__()
        self.imgsz = imgsz
        
        self.encoder = None
        self.decoder = None
        self.fc = nn.Sequential([
            
        ])
        
    def forward(self, x):
        return

######################################################################
# AutoEncoder with Attention Layers                                  #
######################################################################
class attnEncoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(attnEncoder, self).__init__()
        
    def forward(self, x):
        return
    
class attnDecoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(attnDecoder, self).__init__()
        
    def forward(self, x):
        return
    
class attnAutoEncoder(nn.Module):
    def __init__(self, imgsz:tuple):
        super(attnAutoEncoder, self).__init__()
        self.imgsz = imgsz
        
        self.encoder = None
        self.decoder = None
        self.fc = nn.Sequential([
            
        ])
    
    def forward(self, x):
        return
    
if __name__ == '__main__':
    # Test getFlat
    a = torch.randn((16, 3, 224, 224))  #  batched image
    b = torch.randn((3, 224, 224))
    
    aa = getFlat(a)
    bb = getFlat(b)
    
    # print(aa.shape, bb.shape)