import json
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2
import random

def get_CelebA_data(root_folder):
    img_list = os.listdir(os.path.join(root_folder, 'CelebA-HQ-img'))
    label_list = []
    f = open(os.path.join(root_folder, 'CelebA-HQ-attribute-anno.txt'), 'r')
    num_imgs = int(f.readline()[:-1])
    attrs = f.readline()[:-1].split(' ')
    for idx in range(num_imgs):
        line = f.readline()[:-1].split(' ')
        label = line[2:]
        label = list(map(int, label))
        label_list.append(label)
    f.close()
    return img_list, label_list


class CelebALoader(data.Dataset):
    def __init__(self, root_folder, trans=None, cond=False, mode='train'):
        self.root_folder = root_folder
        self.mode = mode
        assert os.path.isdir(self.root_folder), '{} is not a valid directory'.format(self.root_folder)
        
        self.cond = cond
        self.img_list, self.label_list = get_CelebA_data(self.root_folder)
        self.num_classes = 40
        # self.img_list_test = self.img_list[:32]
        # self.label_list_test = self.label_list[:32]
        print("> Found %d images..." % (len(self.img_list)))

        self.transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((64,64)),transforms.ToTensor()])

    def __len__(self):
        return len(self.label_list)


    def __getitem__(self, idx):
        # if self.mode == 'train':
        image = cv2.imread(self.root_folder+'/CelebA-HQ-img/'+self.img_list[idx])[:,:,[2,1,0]]
        Cond = self.label_list[idx]
        image_tensor = self.transform(image).float()
        Cond = np.expand_dims(Cond, 1)
        Cond = np.expand_dims(Cond, 2)
        # else:
        #     image_test = cv2.imread(self.root_folder+'/CelebA-HQ-img/'+self.img_list_test[idx])[:,:,[2,1,0]]
        #     Cond = self.label_list_test[idx]
        #     image_tensor = self.transform(image_test).float()
        #     Cond = np.expand_dims(Cond, 1)
        #     Cond = np.expand_dims(Cond, 2)

        sample = {"Image": image_tensor, "cond": Cond}
        return sample