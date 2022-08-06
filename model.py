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
class basicConv(nn.Module):
    def __init__(self, in_channel, out_channel, \
                    kernel_size, stride, padding):
        """
            기본적인 컨볼루션 레이어입니다. Activation으로 LeakyReLU를 사용합니다.  
            stride = 2일 경우 해상도가 절반으로 감소합니다. 왜그럴까요?
        """
        super(basicConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channel, out_channel, kernel_size, stride, padding
        )
        self.bn = nn.BatchNorm2d(
            out_channel
        )
        self.activation = nn.LeakyReLU(
            0.01, 
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class convEncoder(nn.Module):
    def __init__(self, in_channel):
        super(convEncoder, self).__init__()
        # conv layer
        self.conv1 = basicConv(in_channel, in_channel * 2, 3, 2, 1)
        self.conv2 = basicConv(in_channel * 2, in_channel * 4, 3, 2, 1)
        self.conv3 = basicConv(in_channel * 4, in_channel * 8, 3, 2, 1)
        self.conv4 = basicConv(in_channel * 8, in_channel * 16, 3, 2, 1)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class basicConvT(nn.Module):
    def __init__(self, in_channel, out_channel, \
                    kernel_size, stride, padding):
        super(basicConvT, self).__init__()
        self.convT = nn.ConvTranspose2d(
            in_channel, out_channel, kernel_size, stride, padding
        )
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation = nn.LeakyReLU(
            0.01, 
        )
        
        
    def forward(self, x):
        x = self.convT(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class convDecoder(nn.Module):
    def __init__(self, out_channel):
        super(convDecoder, self).__init__()
        # convT layers
        self.convT1 = basicConvT(out_channel, out_channel // 2, 3, 2, 1)
        self.convT2 = basicConvT(out_channel // 2, out_channel // 4, 3, 2, 1)
        self.convT3 = basicConvT(out_channel // 4, out_channel // 8, 3, 2, 1)
        self.convT4 = basicConvT(out_channel // 8, out_channel // 16, 3, 2, 1)
        
    def forward(self, x):
        x = self.convT1(x)
        x = self.convT2(x)
        x = self.convT3(x)
        x = self.convT4(x)
        return x
    
class convAutoEncoder(nn.Module):
    def __init__(self):
        super(convAutoEncoder, self).__init__()
        self.encoder = convEncoder(3)
        self.decoder = convDecoder(48)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
    
    # Test bassicConv
    a = torch.randn((1, 3, 224, 224))
    convlayer = basicConv(3, 32, 3, 2, 1)
    b = convlayer(a)
    print(b.shape)
    
    # Test convEncoder
    a = torch.randn((1, 3, 224, 224))
    encoder = convEncoder(3)
    b = encoder(a)
    print(b.shape)
    
    # Test basicConvT
    a = torch.randn((1, 48, 14, 14))
    convTlayer = basicConvT(48, 24, 3, 2, 1)
    b = convTlayer(a)
    print(b.shape)