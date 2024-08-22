import pandas as pd

import torch
import torch.nn as nn
from accelerate import Accelerator

from src.nets import *
from src.data import *
from src.agents import *
from src.losses import *

pretrain_epochs = 2
train_epochs = 100
batch_size = 4 * torch.cuda.device_count()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# acc = Accelerator()
# device = acc.device

generator = Generator(scale=2)
discriminator = Discriminator()
vgg = VGG19()
vgg = vgg.to(device)
vgg.eval()

optimizers = {
    'G': torch.optim.Adam(params=generator.parameters(), lr=0.001),
    'D': torch.optim.Adam(params=discriminator.parameters(), lr=0.001),
}

criterions = {
    'MSE': nn.MSELoss(),
    'BCE': nn.BCELoss(),
    'TV': TVLoss().to(device),
    'VGG': PerceptualLoss(vgg).to(device),
    'coeffs': {
        'MSE': 1.0,
        'BCE': 1e-3,
        'TV': 0.0,
        'VGG': 0.006
    }
}

model = SRGANAgent(generator, discriminator, 'logs/SRGAN_v0', device, optimizers, criterions)

train_split = pd.read_csv('data/train.csv', index_col=False)
valid_split = pd.read_csv('data/valid.csv', index_col=False) 
# test_split = pd.read_csv('datasets/mapbox_test.csv', index_col=False)

high_resolution = 5
low_resolution = 10

train_dataset = SRDataset(train_split, low_resolution, high_resolution)
train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              num_workers=2)

valid_dataset = SRDataset(valid_split, low_resolution, high_resolution)
valid_loader = DataLoader(valid_dataset,
                             batch_size=batch_size,
                             pin_memory=True,
                             drop_last=True,
                             num_workers=2)

model.train(pretrain_epochs, train_epochs, train_loader, valid_loader, batch_size, save_frequency=1)