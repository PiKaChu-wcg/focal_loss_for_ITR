r'''
Author       : PiKaChu_wcg
Date         : 2022-01-13 17:57:55
LastEditors  : PiKachu_wcg
LastEditTime : 2022-03-14 10:25:27
FilePath     : /school/VSE_Pytorch/model.py
'''
import timm
# coding: utf-8
import torch
from torch import nn
from utils import l2norm
import random

class ImgSenRanking(torch.nn.Module):
    def __init__(self, model_options):
        super(ImgSenRanking, self).__init__()
        self.model_options = model_options

        self.net=timm.create_model('mobilenetv3_large_100',pretrained=True)
        self.net.classifier=torch.nn.Identity()
        self.linear = torch.nn.Linear(model_options['dim_image'], model_options['dim'],bias=False)

        self.embedding = torch.nn.Embedding(model_options['n_words'], model_options['dim_word'])
        self.lstm = torch.nn.GRU(
            model_options['dim_word'], model_options['dim_word'], 1)
        self.linear2 = torch.nn.Linear(
            model_options['dim_word'], model_options['dim'],bias=False)
        
    def forward(self, x, im,mask):
        return self.forward_sens(x,mask), self.forward_imgs(im)

    def forward_sens(self, x,mask):
        # print(x.shape,mask.shape)
        x_emb = self.embedding(x)

        x_h, _ = self.lstm(x_emb.transpose(1,0))
        #(s,b,d)
        x_h = x_h*mask.transpose(
            0, 1).unsqueeze(-1)
        #(b,d)
        x_h=x_h.sum(0)
        #(b,d) /(b,1)
        x_h=x_h/mask.sum(1, keepdim=True)
        
        x_h = self.linear2(x_h)
        return l2norm(x_h)

    def forward_imgs(self, im):
        im = self.net(im)
        # im=self.drop(im)
        im = self.linear(im)
        return l2norm(im)

    def activate(self):
        for v in self.net.parameters():
            v.requires_grad=True
            
    def freeze(self):
        for v in self.net.parameters():
            v.requires_grad=False
