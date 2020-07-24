# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:21:20 2020

@author: Leibniz
"""

import torch
import torchvision.transforms as transforms

from PIL import Image

import cv2 as cv

class Transfer(object):
    def __init__(self):
        pass
        
        
    def PIL2Tensor(self,src):
        loader=transforms.Compose([transforms.ToTensor()])
        if type(src)==str:
            img=Image.open(src).convert('RGB')
        else:
            img=src
        img=loader(img).unsqueeze(0)
        return img
    
    def Tensor2PIL(self,tensor):
        unloader=transforms.ToPILImgae()
        img=tensor.cpu().clone()
        img=img.squeeze(0)
        img=unloader(img)
        return img
        
    def npy2Tensor(self,src):
        if type(src)==str:
            img=cv.imread(src)
        else:
            img=src
        #assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
        img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        img=torch.from_numpy(img.transpose((2,0,1)))
        return img.float().div(255).unsqueese(0)
    
    def Tensor2npy(self,tensor):
        img=tensor.mul(255).byte()
        img=img.cpu().numpy().squeeze(0).transpose((1,2,0))
        return img
    
    
        
        
    
        