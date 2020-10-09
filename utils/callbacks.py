import os

import torch
from pytorch_lightning import Callback
from pytorch_lightning import Trainer, LightningModule
from torchvision.utils import save_image

from src.models import SRGAN


class LogImages(Callback):
    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        for lr, _, _ in trainer.datamodule.val_dataloader():
            preds = pl_module.netG(lr)
            with open(f'preds{trainer.current_epoch}.png', 'wb+') as f:
                save_image(preds, f, nrow=8)


class CustomCheckpoint(Callback):
    def __init__(self, run_every_e: int = 1, save_last_k: int = 1, save_dir: str = './models/'):
        """
        Saves the most recent k models
        Runs every e epochs
        Args:
            run_every_e (int): How frequently the callback is executed
            save_last_k (int): how many models are saved
            save_dir(str):path of the directory where model
        """
        super().__init__()
        self.run_every_e = run_every_e
        self.save_last_k = save_last_k
        self.save_dir = save_dir

    def on_epoch_end(self, trainer: Trainer, pl_module: SRGAN):
        if trainer.current_epoch % self.run_every_e == 0:
            torch.save(pl_module.netG, os.path.join(self.save_dir,
                                                    f"generator_{trainer.current_epoch}.ckpt"))
            if trainer.current_epoch >= self.save_last_k:
                os.remove(os.path.join(
                    self.save_dir, f"generator_{trainer.current_epoch - self.save_last_k}.ckpt"))
