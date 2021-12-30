import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import sys
import math
import random
import numpy as np
import math
import argparse
import time
import wandb
import matplotlib.pyplot as plt
from torch.nn import init
import os
from dataloader_test import *
from torch.utils.data import Dataset, DataLoader
from glow import Glow
import util

num = 10
test_datasets = CelebALoader('/content/data/task_2/', cond=True)
test_dataloader = DataLoader(test_datasets, batch_size = 1, shuffle = False)

net = Glow(num_channels=512,
               num_levels=4,
               num_steps=6).cuda()

net.load_state_dict(torch.load('/content/gdrive/MyDrive/DL7/Task2/model/image/NF_60.pth'))
net.eval()

sampled_batched = next(iter(test_dataloader))
conds = sampled_batched['cond'].cuda()

def interpolate(z1=None, z2=None):
  split = False
  input_z = torch.zeros((1, 3, 64, 64), dtype=torch.float32).cuda()
  if (z1 == None ) and (z2 == None):
    split = True
    z1 = torch.randn((1, 3, 64, 64), dtype=torch.float32).cuda()
    z2 = torch.randn((1, 3, 64, 64), dtype=torch.float32).cuda()

  input_z = torch.cat([input_z, z1], 0)
  tmp = None

  for i in range(6):
    tmp = torch.lerp(z1, z2, 0.125*(i+1)).cuda()
    input_z = torch.cat([input_z, tmp], 0)

  input_z = torch.cat([input_z, z2], 0)
  input_z = input_z[1:]
  
  if split:
    return input_z[4].unsqueeze(0)
  else:
    return input_z

# task2-1
Z = None
for i in range(conds.shape[0]):
  z = interpolate()
  if i == 0:
    Z = z.clone()
  else:
    Z = torch.cat([Z, z], 0)

img, _ = net(Z.float(), conds.float(), reverse=True)
img = torch.tanh(img)
util.save_image(img, os.path.join('/content/z1.png'), nrow=8, normalize=True)
print('Generate one ...')

# task2-2
for i in range(conds.shape[0]):
  cond = conds[i].unsqueeze(0)
  z_1 = interpolate()
  z_2 = interpolate()
  z = interpolate(z_1, z_2)
  img, _ = net(z.float(), cond.float(), reverse=True)
  img = torch.tanh(img)
  util.save_image(img, os.path.join('/content/z_'+str(i)+'.png'), nrow=8, normalize=True)
  print('Generate one ...')

  



