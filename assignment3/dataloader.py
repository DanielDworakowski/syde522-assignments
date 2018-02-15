from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class npdataset(Dataset):

    def __init__(self, img, label, transform=None):
        self.img = img
        self.label = label
        self.t = transform

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, idx):
        image = self.img[idx, :, :]
        label = self.label[idx]
        sample = {'img': image, 'label': label}
        # 
        # Apply transformations.
        if self.t:
            sample = self.t(sample)
        # 
        # Return the sample.
        return sample