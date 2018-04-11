#!/usr/bin/env python
# coding=utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torchvision.utils import save_image
from torchvision import transforms
from torch.autograd import Variable
from torch.autograd import grad
from model import Generator
from model import Discriminator
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
def to_var(x,volatile=False):
    if torch.cuda.is_available():
        pass#x=x.cuda()
    return Variable(x,volatile=volatile)
def one_hot(lables,dim):
    batch_size=lables.size(0)
    out = torch.zeros(batch_size,dim)
    out[np.arange(batch_size),lables.long()]=1
    return out
def denorm(x):
    out = (x+1)/2
    return out.clamp_(0,1)

G = Generator(64,5+8+2,6)
transforms1 = transforms.Compose([
    transforms.CenterCrop(300),
    transforms.Scale(216,interpolation=Image.ANTIALIAS),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
img = Image.open('9.jpg').convert('RGB')
img2 = transforms1(img)
img2 = img2.unsqueeze(0)
img2=Variable(img2)
#print(img2)
#orc_c = torch.FloatTensor([])

c = torch.FloatTensor([[0,0,0,0,0,0,0,0,0,0,1,0,0,0,1]])
c = Variable(c)
#print(c)
# Load trained parameters
#G.load_state_dict(torch.load('stargan_celebA/models/20_12000_G.pth'))
G.load_state_dict(torch.load('stargan_both/models/350000_G.pth'))
G.eval()
real_x = img2
target_c2_list = []
for j in range(8):
    target_c =one_hot(torch.ones(real_x.size(0)) * j, 8)
    target_c2_list.append(to_var(target_c, volatile=True))
# Zero vectors and mask vectors
zero1 = to_var(torch.zeros(real_x.size(0), 8))     # zero vector for rafd expressions
mask1 = to_var(one_hot(torch.zeros(real_x.size(0)), 2)) # mask vector: [1, 0]
zero2 = to_var(torch.zeros(real_x.size(0), 5))      # zero vector for celebA attributes
mask2 = to_var(one_hot(torch.ones(real_x.size(0)), 2))  # mask vector: [0, 1]

fake_image_list = [real_x]
for j in range(8):
    target_c = torch.cat([zero2,target_c2_list[j],mask2],dim=1)
    temp = G(real_x,target_c)
    print(temp.size())
    fake_image_list.append(temp)
img_output = G(img2,c)
save_path = 'output_fake.png'
fake_images = torch.cat(fake_image_list, dim=3)
save_image(denorm(fake_images.data), save_path, nrow=1, padding=0)
print('Translated test images and saved into "{}"..!'.format(save_path))
img_output = denorm(img_output.data)
save_image(img_output,'output.jpg',nrow=1,padding=0)
