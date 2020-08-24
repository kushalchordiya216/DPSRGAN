import os
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from networks import Generator, Discriminator
from losses import ContentLoss, DownScaleLoss

content_loss = ContentLoss()
downscale_loss = DownScaleLoss()


class PreTrainGenModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.netG = Generator()
        self.netG.cuda()

    def forward(self, x):
        return self.netG(x)

    def training_step(self, batch, batch_idx):
        lr, hr, _ = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr) + content_loss(sr, hr)

        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return result

    def validation_step(self, batch, batch_idx):
        lr, hr, _ = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr) + content_loss(sr, hr)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def configure_optimizers(self):
        return Adam(self.netG.parameters(), lr=0.0002)


class SRGAN(pl.LightningModule):
    def __init__(self, hparams):
        super(SRGAN, self).__init__()
        self.hparams = hparams

        self.netG: nn.Module = Generator()
        self.netD: nn.Module = Discriminator()

        self.generated_imgs = None
        self.last_imgs = None

    def forward(self, z):
        return self.netG(z)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb, optimizer_idx):
        lr, hr, interpolated_lr = batch
        self.last_imgs: Tensor = hr
        self.generated_imgs: Tensor = self(lr)
        if optimizer_idx == 1:
            real = torch.ones((hr.size(0), 1, 5, 5), device=self.device)

            # with torch.no_grad():
            D_fake: Tensor = self.netD(self.generated_imgs, interpolated_lr)

            g_loss = 0.001 * self.adversarial_loss(D_fake, real) + content_loss(self.generated_imgs, hr)

            if self.hparams.downscale_loss:
                g_loss += downscale_loss(self.generated_imgs, lr)

            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        elif optimizer_idx == 0:
            print("Disc train")
            real = 0.3 * torch.rand((hr.size(0), 1, 5, 5), device=self.device) + 0.7
            fake = 0.3 * torch.rand((hr.size(0), 1, 5, 5), device=self.device)
            # label smoothing, between 0.7-1.0 for real and 0.0 to 1.2 for fake

            real_loss = self.adversarial_loss(self.netD(hr, interpolated_lr), real)
            fake_loss = self.adversarial_loss(self.netD(self.generated_imgs.detach(), interpolated_lr), fake)

            d_loss = (fake_loss + real_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def validation_step(self, batch, batch_idx):
        lr, hr, interpolated_lr = batch
        sr = self(lr)
        D_fake = self.netD(sr, interpolated_lr)
        real = torch.ones((hr.size(0), 1, 5, 5), device=self.device)

        val_loss = self.adversarial_loss(D_fake, real) + content_loss(sr, hr)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def configure_optimizers(self):
        opt_g = Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-8)
        opt_d = Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-8)
        return [opt_d, opt_g], []  # second array is for lr schedulers if needed
