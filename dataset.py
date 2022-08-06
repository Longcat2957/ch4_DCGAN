import os
import random
import torch
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as T

from PIL import Image
import numpy as np

def readImg(imgPath:str):
    return Image.open(imgPath)

def getImglist(imgsDir:str):
    return os.listdir(imgsDir)

# Anomaly Detection Dataset Custom Dataset Class
class BTechDataset(Dataset):
    """BTechDataset Class, torch.utils.data.Dataset 클래스를 상속받았다.

    Args:
        rootDir : '../datasets/BTech_Dataset_transformed'
        clsn : 1, 2, 3
        phase : 'train' or 'test'
        type : 'ok' (train) or 'ok'/'ko' (test)
        
    Specification:
        self.transformer include torchvision.transforms.Normalize(<ImageNet Constants>)
        output tensor shape = (3, 224, 224)
    """
    def __init__(self, rootDir:str, clsn:int, phase:str='train', type:str='ok'):
        self.imgSize = (224, 224)
        self.imgsDir = os.path.join(rootDir, '0' + str(clsn), phase, type)
        self.imgsList = getImglist(self.imgsDir)
        self.transformer = T.Compose([
            T.Resize(size=self.imgSize),
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.imgsList)
    
    def __getitem__(self, idx):
        imgPath = os.path.join(self.imgsDir, self.imgsList[idx])
        origImg = readImg(imgPath)
        return self.transformer(origImg)


if __name__ == '__main__':
    BTechRootPath = '../datasets/BTech_Dataset_transformed'
    btech_train = BTechDataset(
        BTechRootPath,
        1,
        'train',
        'ok'
    )
    btech_val = BTechDataset(
        BTechRootPath,
        1,
        'test',
        'ko'
    )
    
    # length of dataset
    print(f'btech_train : {len(btech_train)}, btech_val : {len(btech_val)}')
    
    # get one of dataset index
    train_idx = random.randint(0, len(btech_train) - 1)
    val_idx = random.randint(0, len(btech_val) - 1)
    print(f'train_idx, val_idx = {train_idx}, {val_idx}')
    
    # get one of data
    train = btech_train[train_idx]
    val = btech_val[val_idx]
    print(f'shape of train, val data = {train.shape}, {val.shape}')