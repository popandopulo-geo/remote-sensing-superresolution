import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import albumentations as A
from skimage.transform import rescale, resize
from PIL import Image
import os

class SRDataset(Dataset):
    def __init__(self, dataframe, low_resolution=10, high_resolution=5, transforms=None):
        super(SRDataset).__init__()
        resolution = 0.3
        
        self.df = dataframe
        
        self.lr_scale = resolution / low_resolution
        self.hr_scale = resolution / high_resolution
        
        self.add_transforms = transforms
        self.base_transforms = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        self.crop_shape = (2134, 2134)
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        raster = rasterio.open(self.df.iloc[idx]['tiff_image_path'])
        image = raster.read().transpose((1,2,0))
        
        transformed = self.base_transforms(image=image)
        image = transformed['image']
        
        image = torch.tensor(image).permute((2,0,1))
        i, j, h, w = transforms.RandomCrop.get_params(image, self.crop_shape)
        image = F.crop(image, i, j, h, w)
        image = image.numpy().transpose((1,2,0))
        
        lr = (rescale(image, (self.lr_scale, self.lr_scale, 1)) / 0.5) - 1
        hr = (rescale(image, (self.hr_scale, self.hr_scale, 1)) / 0.5) - 1
        
        lr = torch.tensor(lr, dtype=torch.float32).permute((2,0,1))
        hr = torch.tensor(hr, dtype=torch.float32).permute((2,0,1))

        if self.add_transforms is None:
            return {'LR': lr,
                    'GT': hr}
        else:
            pass