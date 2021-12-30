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

test_datasets = CelebALoader('/content/data/task_2/', cond=True)
num = len(train_test)
test_dataloader = DataLoader(train_datasets, batch_size = num, shuffle = False)

net = Glow(num_channels=512,
               num_levels=4,
               num_steps=6).cuda()

net.load_state_dict(torch.load('/content/gdrive/MyDrive/DL7/Task2/model/image/NF_50.pth'))
net.eval()

sampled_batched = next(iter(test_dataloader))
conds = sampled_batched['cond'].cuda()
inputs = sampled_batched['Image'].cuda()

# get attribute latent
z_pos = 0
for i in range(num-1):
  cond = conds[i].unsqueeze(0)
  input_ = inputs[i].unsqueeze(0)
  z_, _ = net(input_.float(), cond.float(), reverse=False)
  z_pos += z_

z_pos = z_pos/(num-1)


# addd to target image (no smile)
cond = conds[-1].unsqueeze(0)
input_ = inputs[-1].unsqueeze(0)
z_in, _ = net(input_.float(), cond.float(), reverse=False)

Input = torch.zeros((1, 3, 64, 64), dtype=torch.float32).cuda()
Input = torch.cat([Input, z_in], 0)
for i in range(4):
  tmp = z_in + 0.25*(i+1)*z_pos
  Input = torch.cat([Input, tmp], 0)
Input = Input[1:]
img, _ = net(Input, cond.float(), reverse=True)
util.save_image(img, os.path.join('/content/z3.png'), nrow=8, normalize=True)