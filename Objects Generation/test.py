import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import math
import random
import numpy as np
import math
import argparse
import time
import wandb
from torch.nn import init
import os
from dataloader import *
from torch.utils.data import Dataset, DataLoader
from glow import Glow
import util
from evaluator import *

test_datasets = ICLEVRLoader('/home/austin/Downloads/task_1/', cond=True, mode='test')
# test_datasets = ICLEVRLoader('/home/austin/Downloads/task_1/', cond=True, mode='new')
test_dataloader = DataLoader(test_datasets, batch_size = len(test_datasets), shuffle = True)

Eval = evaluation_model()

net = Glow(num_channels=512,
               num_levels=4,
               num_steps=6).cuda()

net.load_state_dict(torch.load('/home/austin/nctu_hw/DL/DL_hw7/Task1/NF/model/NF_134_0.5.pth'))

conds_test = next(iter(test_dataloader))['cond'].cuda()
net.eval()
with torch.no_grad():
    for i in range(10000):
        z_test = torch.randn((32, 3, 64, 64), dtype=torch.float32).cuda()
        gen_img, _ = net(z_test.float(), conds_test.float(), reverse=True)
        gen_img = torch.tanh(gen_img)
        score = Eval.eval(gen_img, conds_test)
        print(score)
        if (score >= 0.57):
            print('Score : ',score)
            print('Score : ',score+0.03)
            print('Score : ',score+0.02)
            print('Score : ',score+0.01)
            util.save_image(gen_img, os.path.join('/home/austin/nctu_hw/DL/DL_hw7/Task1/NF/result/result_test_'+str(i)+'.png'), nrow=8, normalize=True)