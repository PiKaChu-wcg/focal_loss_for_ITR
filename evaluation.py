r'''
Author       : PiKaChu_wcg
Date         : 2022-01-28 00:53:29
LastEditors  : PiKachu_wcg
LastEditTime : 2022-01-31 01:16:58
FilePath     : /VSE_Pytorch/evaluation.py
'''
import numpy as  np
import torch
from datasets import load_dataset
from tools_ import encode_sentences, encode_images

def evalrank(model, data, split='dev'):
    """
    Evaluate a trained model on either dev ortest
    """

    print('Loading dataset')
    if split == 'dev':
        X = load_dataset(data)[1]
    else:
        X = load_dataset(data, load_test=True)


    print('Computing results...')
    ls = encode_sentences(model, X[0])
    lim = encode_images(model, X[1])

    eval(lim,ls)

# todo img input is N instead of 5N 
# todo speedify 
def i2t(dis,n):
    inds=torch.sort(dis,descending=True,dim=1)[1].sort(dim=1)[1]
    ranks=inds.reshape(n,n,5).min(dim=-1)[0].diag()
    # Compute metrics
    r1 = (100.0 * (ranks<1).sum() / ranks.numel()).item()
    r5 =  (100.0 * (ranks<5).sum() / ranks.numel()).item()
    r10 = (100.0 * (ranks < 10).sum() / ranks.numel()).item()
    medr = int(ranks.median().item())+1
    return (r1, r5, r10, medr)
 
  
def t2i(dis, n):
    inds = torch.sort(dis, descending=True, dim=0)[1].sort(dim=0)[1]
    inds = inds.reshape(n, n, 5)
    ranks = torch.concat([inds[:, :, i].diag() for i in range(5)])
    # Compute metrics
    r1 = (100.0 * (ranks < 1).sum() / ranks.numel()).item()
    r5 = (100.0 * (ranks < 5).sum() / ranks.numel()).item()
    r10 = (100.0 * (ranks < 10).sum() / ranks.numel()).item()
    medr = int(ranks.median().item())+1
    return (r1, r5, r10, medr)


def eval(x, img, writer=None,epoch=None,repeated=False,verbo=False):
    if verbo:
        print("\033[32mComputing the score!\033[0m")
    if repeated:
        img=img[::5,:]
    n = img.shape[0]
    dis = torch.mm(img, x.t())
    (r1, r5, r10, medr) = i2t(dis, n)
    (r1i, r5i, r10i, medri) = t2i(dis,n)
    if verbo:
        print("the eval result: R@1   R@5   R@10  R@med")
        print("Image to text:   %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr))
        print("Text to image:   %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri))
        if writer is not None:
            writer.add_scalar('i2t/R@1',r1,epoch)
            writer.add_scalar('i2t/R@5',r5,epoch)
            writer.add_scalar('i2t/R@10', r10,epoch)
            writer.add_scalar('i2t/R@med', medr,epoch)
            writer.add_scalar('t2i/R@1', r1i,epoch)
            writer.add_scalar('t2i/R@5', r5i,epoch)
            writer.add_scalar('t2i/R@10', r10i,epoch)
            writer.add_scalar('t2i/R@med', medri,epoch)
    return (r1, r5, r10, medr),(r1i, r5i, r10i, medri)