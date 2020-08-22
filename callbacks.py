import torch
from pytorch_lightning import Callback
from dataloader import SRDataLoader
from torchvision.utils import make_grid, save_image
from pytorch_lightning import Trainer
data = SRDataLoader(data_dir='images/*.jpg')
data.setup()


class LogImages(Callback):
    def on_epoch_end(self, trainer: Trainer, pl_module):
        for lr, hr, _ in data.val_dataloader():
            preds = pl_module.netG(lr)

            with open(f'preds.png', 'wb+') as f:
                save_image(preds, f, nrow=8)
            with open(f'input.png', 'wb+') as f:
                save_image(lr, f, nrow=8)
