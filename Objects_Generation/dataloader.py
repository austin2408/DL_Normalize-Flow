import json
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2
import random

def get_iCLEVR_data(root_folder,mode):
    if mode == 'train':
        data = json.load(open(os.path.join(root_folder,'train.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        img = list(data.keys())
        label = list(data.values())
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return np.squeeze(img), np.squeeze(label)
        
    else:
        if mode == 'test':
            print('get test data')
            data = json.load(open(os.path.join(root_folder,'test.json')))
        else:
            print('get new data')
            data = json.load(open(os.path.join(root_folder,'new.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        label = data
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
            
        return None, label

class ICLEVRLoader(data.Dataset):
    def __init__(self, root_folder, trans=None, cond=False, mode='train'):
        self.root_folder = root_folder
        self.mode = mode
        self.img_list, self.label_list = get_iCLEVR_data(root_folder,mode)
        if self.mode == 'train':
            print("> Found %d images..." % (len(self.img_list)))
        
        # self.cond = cond
        # self.num_classes = 24
        self.transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((64,64)),transforms.ToTensor()])
        
        
                
    def __len__(self):
        """'return the size of dataset"""
        return len(self.label_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            image = cv2.imread(self.root_folder+'/images/'+self.img_list[idx])[:,:,[2,1,0]]
            self.shape = image.shape
            Cond = self.label_list[idx]
            image_tensor = self.transform(image).float()
            Cond = np.expand_dims(Cond, 1)
            Cond = np.expand_dims(Cond, 2)

            sample = {"Image": image_tensor, "cond": Cond}
        else:
            Cond = self.label_list[idx]
            Cond = np.expand_dims(Cond, 1)
            Cond = np.expand_dims(Cond, 2)

            sample = {"cond": Cond}

        return sample


