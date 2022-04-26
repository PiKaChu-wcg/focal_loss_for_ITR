r'''
Author       : PiKaChu_wcg
Date         : 2022-02-10 19:28:37
LastEditors  : PiKachu_wcg
LastEditTime : 2022-02-11 09:59:24
FilePath     : /school/VSE_Pytorch/callback.py
'''
from curses.ascii import BS
from pytorch_lightning.callbacks.base import Callback

class FineTune(Callback):
    def __init__(self,epoch,logger,bs,hard):
        super(FineTune, self).__init__()
        self.epoch=epoch
        self.logger=logger
        self.bs=bs
        self.hard=hard
    def on_epoch_end(self,trainer,pl_module):
        # self.logger.info(f"{trainer.current_epoch}")
        if trainer.current_epoch==self.epoch:
            trainer.train_dataloader.dataset.hard=self.hard
            self.logger.info("activate the visual encode")