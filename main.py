r'''
Author       : PiKaChu_wcg
Date         : 2022-01-29 21:26:49
LastEditors  : PiKachu_wcg
LastEditTime : 2022-04-15 10:48:41
FilePath     : /school/VSE_Pytorch/main.py
'''
import argparse
import warnings
import os

import pytorch_lightning as pl
import torch
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer

from callback import *
from datasets import *
from evaluation import eval
from loss import *
from model import *

os.chdir('/data/wuchangguang/school/VSE_Pytorch/')
warnings.filterwarnings("ignore")
opts = dict(
    vocab_path='./vocab/',
    dim_image=1280,
    dim=512,
    dim_word=256,
    margin=0.2,
    epochs=30,
    batch_size=64,
    lrate=0.001,
    gpus=[0,1,2,3],
    detach=False
)
# opts['loss']=InfoNCE(0.1,learnable=True)
opts['loss'] = FocalInfoNCE(0.1,alpha=0.5,gamma=1.5,learnable=True)
# opts['loss'] = PairwiseRankingLoss(opts['margin'])
# opts['loss'] = FocalTriplet(opts,gamma=1.5,margin=0.4,learnable=True)


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hard', action='store_true')
    parser.add_argument('--batch_size', default=opts['batch_size'], type=int)
    args = vars(parser.parse_args())
    for i, k in args.items():
        opts[i] = k
    print(opts)


class MyModel(pl.LightningModule):
    def __init__(self, opts):
        super(MyModel, self).__init__()
        self.opts = opts
        self.tokenizer = self.prepara_tokenizer()
        self.model = ImgSenRanking(self.opts)
        self.loss_model = opts['loss']

    def prepara_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.opts['vocab_path'], model_max_lengt=128)
        n_words = len(tokenizer.get_vocab())
        self.opts['n_words'] = n_words
        return tokenizer

    def forward(self, im, txt, mask):
        txt, im = self.model(txt, im, mask)
        txt=self.all_gather(txt,sync_grads=True).flatten(0,1)
        im=self.all_gather(im,sync_grads=True).flatten(0,1)  
        loss = self.loss_model(im, txt)
        return loss

    def training_step(self, batch, batch_idx):
        (txt, mask), im = batch
        loss = self(im, txt, mask)
        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (txt, mask), im = batch
        txt = self.model.forward_sens(txt, mask)
        im = self.model.forward_imgs(im)
        return txt, im
    # def validation_step_end(self, batch_parts):

    #     TXT = torch.concat(batch_parts['txt'], dim=0)
    #     IM = torch.concat(batch_parts['im'], dim=0)
    #     return TXT,IM
    def validation_epoch_end(self, validation_step_outputs):
        TXT = []
        IM = []
        for txt, im in validation_step_outputs:
            TXT.append(txt)
            IM.append(im)
        TXT = torch.concat(TXT, dim=0)
        IM = torch.concat(IM, dim=0)
        TXT = self.all_gather(TXT, sync_grads=True).flatten(0, 1)
        IM = self.all_gather(IM, sync_grads=True).flatten(0, 1)

        (r1, r5, r10, medr), (r1i, r5i, r10i, medri) = eval(TXT, IM)
        i2t = {
            "i2t/R@1": r1,
            "i2t/R@5": r5,
            "i2t/R@10": r10,
            "i2t/R@med": medr

        }
        t2i = {
            "t2i/R@1": r1i,
            "t2i/R@5": r5i,
            "t2i/R@10": r10i,
            "t2i/R@med": medri
        }
        self.log_dict(i2t)
        self.log_dict(t2i)
        return (r1, r5, r10, medr), (r1i, r5i, r10i, medri)
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_step_end(self, batch_parts):
        return self.validation_step_end(batch_parts)

    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.opts['lrate'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.opts['epochs'], self.opts['lrate']*0.01)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    get_opts()

    model = MyModel(opts)

    train, dev,test = Flickr30K(split='coco'), Flickr30K(split='val'),Flickr30K(split='test')
    train_loader, train_dataset = get_loader(
        train, model.tokenizer, batch_size=opts['batch_size'])
    val_loader, val_dataset = get_loader(
        dev, model.tokenizer, batch_size=opts['batch_size']*2, shuffle=False)
    test_loader, test_dataset = get_loader(
        test, model.tokenizer, batch_size=opts['batch_size']*2, shuffle=False)
    # test = load_dataset(name=opts.dataset, load_test=True)
    # test_loader, test_dataset = get_loader(
    #     test, model.tokenizer, batch_size=1, shuffle=False)
    ckpt_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=10,
        monitor='i2t/R@1',
        mode='max'
    )
    trainer = pl.Trainer(
        gpus=opts['gpus'],
        strategy="ddp_find_unused_parameters_false",
        # precision=16,
        # amp_backend='apex',
        gradient_clip_val=2,
        callbacks=[ckpt_callback],
        max_epochs=opts['epochs']
    )
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=[val_loader])
    res=trainer.test(model,dataloaders=[test_loader])[0]
    RANK = int(os.getenv('LOCAL_RANK', -1))
    if RANK in [0,-1]:
        for i in res.keys():
            res[i]=[res[i]]
        df=pd.read_csv('../results.csv')
        df=df.append(pd.DataFrame(res))
        df.to_csv('../results.csv',index=False)