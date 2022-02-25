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

parser = argparse.ArgumentParser(description='setup')
parser.add_argument('--data', default='/home/austin/Downloads/task_1/', type=str)
parser.add_argument('--store', default='/home/austin/nctu_hw/DL/DL_hw7/Task1/NF/model/', type=str)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--num_channels', default=512, type=int)
parser.add_argument('--num_levels', default=4, type=int)
parser.add_argument('--num_steps', default=6, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--epochs', default=500, type=int)
args = parser.parse_args()
print(args)


train_datasets = ICLEVRLoader(args.data, cond=True)
train_dataloader = DataLoader(train_datasets, batch_size = args.batch_size, shuffle = True)

test_datasets = ICLEVRLoader(args.data, cond=True, mode='test')
test_dataloader = DataLoader(test_datasets, batch_size = len(test_datasets), shuffle = True)



net = Glow(num_channels=args.num_channels,
               num_levels=args.num_levels,
               num_steps=args.num_steps).cuda()

optimizer = optim.Adam(net.parameters(), lr=args.lr)

Eval = evaluation_model()
criterion = util.NLLLoss().cuda()
loss_meter = util.AverageMeter()

wandb.init(project='NF-task1')
config = wandb.config
config.lr = args.lr
config.batch_size = args.batch_size
config.num_channels = args.num_channels
config.num_levels = args.num_levels
config.num_steps = args.num_steps
config.epochs = args.epochs
wandb.watch(net)


z_test = torch.randn((32, 3, 64, 64), dtype=torch.float32).cuda()
conds_test = next(iter(test_dataloader))['cond'].cuda()

for epoch in range(args.epochs):
    loss_sum = []
    score = 0
    net.train()
    for i_batch, sampled_batched in enumerate(train_dataloader):
        print(str(i_batch) + '/'+str(int(len(train_datasets)/args.batch_size)+1),end='\r')

        inputs = sampled_batched['Image'].cuda()
        conds = sampled_batched['cond'].cuda()

        optimizer.zero_grad()

        z, sldj = net(inputs.float(), conds.float(), reverse=False)

        loss = criterion(z, sldj)
        loss_meter.update(loss.item(), z.size(0))
        loss.backward()

        optimizer.step()

        loss_sum.append(0.0001*loss.item())
    
    net.eval()
    with torch.no_grad():
        gen_img, _ = net(z_test.float(), conds_test.float(), reverse=True)
        gen_img = torch.tanh(gen_img)
        score = Eval.eval(gen_img, conds_test)

    print('Epoch : '+str(epoch+1)+' Loss : '+str(sum(loss_sum)/len(loss_sum))+' Score : '+str(score))

    wandb.log({"Loss": sum(loss_sum)/len(loss_sum)})
    wandb.log({"Score": score})

    if (score>=0.5):
        util.save_image(gen_img, os.path.join(args.store, f'2epoch{epoch+1}.png'), nrow=8, normalize=True)
        torch.save(net.state_dict(), args.store+'NF_'+str(epoch+1)+'_'+str(score)+'.pth')




