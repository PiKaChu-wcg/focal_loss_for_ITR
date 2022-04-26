r'''
Author       : PiKaChu_wcg
Date         : 2022-03-14 10:17:59
LastEditors  : PiKachu_wcg
LastEditTime : 2022-04-08 16:31:49
FilePath     : /school/VSE_Pytorch/loss.py
'''
import timm
# coding: utf-8
import torch
from torch import nn
from utils import l2norm
import random

class PairwiseRankingLoss(torch.nn.Module):

    def __init__(self, margin=0.2,hard=False):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin
        self.hard=hard

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = im.mm(s.t())
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)
        cost_s = cost_s-cost_s.diag().diag_embed()
        cost_im = cost_im-cost_im.diag().diag_embed()
        # I = torch.eye(scores.size(0)) > .5
        # if torch.cuda.is_available():
        #     I = I.to(scores.device )
        # cost_s = cost_s.masked_fill_(I, 0)
        # cost_im = cost_im.masked_fill_(I, 0)
        # if random.random()<0.003:
        #     with open('log.txt','w') as f:
        #         diagonal.requires_grad_(True)
        #         cost_s.max(0)[0].sum().backward()
                
        #         f.write(f"{diagonal.grad}")
        if self.hard:
            cost_im = cost_im.max(0)[0]
            cost_s = cost_s.max(1)[0]
            return cost_s.sum() + cost_im.sum()
        else:
            return (cost_s.sum()+cost_im.sum())/scores.size(0)


class FocalTriplet(torch.nn.Module):
    def __init__(self,opts,gamma=1.5,temperature=0.1,margin=0.2,learnable=False):
        super(FocalTriplet, self).__init__()
        self.opts=opts
        self.loss_fn=FocalLoss(nn.BCELoss(),alpha=0.9)
        self.gamma=gamma
        self.temperature = temperature if not learnable else nn.Parameter(
            torch.tensor(temperature))
        self.margin=margin
    def forward(self,im,s):
        scores = im.mm(s.t())
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = scores - d1
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im =  scores - d2
        cost_s = (self.margin+cost_s-cost_s.diag().diag_embed()).clamp(min=0)
        cost_im = (self.margin+cost_im-cost_im.diag().diag_embed()).clamp(min=0)

        scores_im=(cost_im/self.temperature).softmax(dim=-1)
        scores_sen=(cost_s/self.temperature).softmax(dim=-2)

        return (scores_im**self.gamma*cost_im+scores_sen**self.gamma*cost_s).sum()/im.size(0)

        # return ((cost_im+cost_s)*(1-sim)**self.gamma).sum()/b+loss

         


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.5):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        # true=torch.nn.functional.one_hot(true)
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class FocalInfoNCE(nn.Module):
    def __init__(self, temperature=0.1, gamma=1.5, alpha=0.5, learnable=False):
        super(FocalInfoNCE,self).__init__()
        self.loss_fn=FocalLoss(nn.BCELoss(),gamma, alpha)
        self.temprature = temperature if not learnable else nn.Parameter(
            torch.tensor(temperature))
    def forward(self,im,s):
        scores = im.mm(s.t())
        bs=scores.size(0)
        self.gt = torch.range(0, bs-1).long()
        self.gt=torch.nn.functional.one_hot(self.gt).float().to(scores.device)
        scores_im=(scores/self.temprature).softmax(dim=-1)
        scores_sen=(scores.T/self.temprature).softmax(dim=-1)
        loss_im=self.loss_fn(scores_im,self.gt)
        loss_sen=self.loss_fn(scores_sen,self.gt)
        return (loss_im+loss_sen)/2

class InfoNCE(nn.Module):
    def __init__(self,temperature=0.1,learnable=False):
        super(InfoNCE,self).__init__()
        self.loss_fn=nn.CrossEntropyLoss()
        self.temprature = temperature if not learnable else nn.Parameter(
            torch.tensor(temperature))

    def forward(self,im,s):
        # print(im.size(),s.size())
        scores=im.mm(s.t())
        bs=scores.size(0)
        self.gt = torch.range(0, bs-1).long().to(scores.device)
        # self.gt=torch.nn.functional.one_hot(self.gt).float()
        scores_im=scores/self.temprature
        scores_sen=scores.T/self.temprature
        return (self.loss_fn(scores_im,self.gt)+self.loss_fn(scores_sen,self.gt))/2