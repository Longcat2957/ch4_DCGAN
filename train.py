import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import time
import numpy as np
from tqdm import tqdm
import copy

def trainAE(model, \
    trainDataloader:DataLoader, testDataloader:DataLoader, \
    criterion, optimizer, lr_scheduler, \
    threshold = 0.95, \
    num_of_epochs = 300):
    """

    Args:
        model (_type_): _description_
        testDataloader (DataLoader): _description_
        optimizer (_type_): _description_
        lr_scheduler (_type_): _description_
    """
    
    
    
    return 