import pandas as pd
import os
import numpy as np
from sklearn.utils import class_weight
import torch

def calc_class_weights(train_df, device):
    class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.array([0,1,2,3,4]),y=train_df['diagnosis'].values)
    class_weights = torch.tensor(class_weights,dtype=torch.float).to(device)
    
    return class_weights
    