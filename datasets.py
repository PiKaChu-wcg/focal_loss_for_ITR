r'''
Author       : PiKaChu_wcg
Date         : 2022-01-13 20:11:09
LastEditors  : PiKachu_wcg
LastEditTime : 2022-04-09 13:12:13
FilePath     : /school/VSE_Pytorch/datasets.py
'''
# todo transformer
from glob import glob
import cv2
import pandas as pd
from transformers import AutoTokenizer
import torch.nn.utils.rnn as rnn_utils
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
import os

def load_dataset(name='f8k', load_test=False, path_to_data='../'):
    loc = path_to_data+name+'/'
    if load_test:
        test_caps = []
        with open(loc+name+'_test_caps.txt', 'r') as f:
            for line in f:
                test_caps.append(line.strip())
        test_ims = np.load(loc+name+"_test_ims.npy")
        return (test_caps, test_ims)
    else:
        train_caps, dev_caps = [], []
        with open(loc+name+'_train_caps.txt', 'r') as f:
            for line in f:
                train_caps.append(line.strip())

        with open(loc+name+'_dev_caps.txt', 'r') as f:
            for line in f:
                dev_caps.append(line.strip())

        # Image features
        train_ims = np.load(loc+name+'_train_ims.npy')
        dev_ims = np.load(loc+name+'_dev_ims.npy')

        return (train_caps, train_ims), (dev_caps, dev_ims)


# class HomogeneousData(Dataset):
#     def __init__(self, data, maxlen=1024):
#         self.sen, self.img = data[0], data[1]
#         self.maxlen = maxlen

#     def __len__(self):
#         return len(self.img)

#     def __getitem__(self, index):
#         return self.sen[index][:self.maxlen]+'[PAD]', self.img[index]


class Flickr30K(Dataset):
    def __init__(self, dir='../flickr30k', split='train',transformer=None):
        self.dir = dir
        self.split = split
        with open(self.dir+f"/flickr30k_{split}.txt", 'r') as f:
            self.image_list = [line.strip() for line in f.readlines()]
        self.df = pd.read_csv(self.dir+'/results.csv',sep='|')
        # print(f"the size of data {self.df.shape[0]}")
        # print(self.df)
        self.transformer=transforms.Compose([
            transforms.ToTensor(),
            transformer,
            transforms.Resize((384,384)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ]
        ) if transformer is not None else transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Resize((384,384)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ]
        )
        
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, index):
        img = cv2.imread(self.dir+"/Images/"+self.image_list[index])
        txt = list(self.df[self.df['image'] ==
                   self.image_list[index]]['caption'])

        assert img is not None, f"{self.image_list[index]},{index}"
        assert len(txt) > 0,  f"{txt},{self.image_list[index]}"
        if self.split in ['train','coco']:
            i = np.random.randint(0, 5)
            txt = txt[i]
        img=self.transformer(img).unsqueeze(0)
        return txt, img


def get_loader(dataset, tokenizer=None, batch_size=128, maxlen=512, n_works=8, shuffle=True):
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english") if not tokenizer else tokenizer
    def collate_fn(batch):
        sen_t, img = list(zip(*batch))
        # print(sen_t,img)
        sen = []
        if isinstance(sen_t[0], list):
            for i in sen_t:
                sen.extend(i)
        else:
            sen = sen_t
        t = tokenizer(list(sen), padding=True, return_tensors="pt")
        sen, mask = t['input_ids'], t['attention_mask']
        # print(img[0].shape,img[1].shape,img[2].shape)
        img = torch.concat(img, dim=0)
        return (sen, mask), img



    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=shuffle, num_workers=n_works, collate_fn=collate_fn,pin_memory=True)
    return loader, dataset
