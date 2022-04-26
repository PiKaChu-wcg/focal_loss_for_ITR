
r'''
Author       : PiKaChu_wcg
Date         : 2022-01-13 17:57:55
LastEditors  : PiKachu_wcg
LastEditTime : 2022-01-14 14:26:12
FilePath     : /school/VSE_Pytorch/utils.py
'''
import numpy as np
from torch.autograd import Variable
import torch

def l2norm(input, p=2.0, dim=1, eps=1e-12):
    """
    Compute L2 norm, row-wise
    """
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)
# def l2norm(X):
#     """L2-normalize columns of X
#     """
#     norm = torch.pow(X, 2).sum(dim=0, keepdim=True).sqrt()
#     X = torch.div(X, norm)
#     return X

def xavier_weight(tensor):
    if isinstance(tensor, Variable):
        xavier_weight(tensor.data)
        return tensor

    nin, nout = tensor.size()[0], tensor.size()[1]
    r = np.sqrt(6.) / np.sqrt(nin + nout)
    return tensor.normal_(0, r)


def sort_by_len(dataset):
    dataset = list(zip(*dataset))
    dataset.sort(key=lambda x: len(x[0]))
    return list(zip(*dataset))
