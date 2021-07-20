from argparse import ArgumentParser
from collections import OrderedDict
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torchvision.utils import save_image

from src.networks import Generator, Discriminator
from src.losses import ContentLoss

content_loss = ContentLoss()


class SRResNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.gpu: bool = torch.cuda.is_available()
        self.netG = Generator()
        if self.gpu:
            self.netG = self.netG.cuda()

    def forward(self, x):
        return self.netG(x)

    def training_step(self, batch: List[Tensor], batch_idx):
        lr, hr, _ = batch
        if self.gpu:
            lr, hr = lr.cuda(), hr.cuda()
        sr = self(lr)
        loss = F.mse_loss(sr, hr) + content_loss(sr, hr)

        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_epoch=True,
                   prog_bar=True, logger=True)
        return result

    def test_step(self, batch: List[Tensor], batch_idx: int):
        lr, _, _ = batch
        sr = self(lr)
        with open(f'preds/{batch_idx}.png', 'wb+') as f:
            save_image(sr, f)
        pass

    def validation_step(self, batch: List[Tensor], batch_idx):
        lr, hr, _ = batch
        if self.gpu:
            lr, hr = lr.cuda(), hr.cuda()
        sr = self(lr)
        loss = F.mse_loss(sr, hr) + content_loss(sr, hr)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def configure_optimizers(self):
        return Adam(self.netG.parameters(), lr=0.0002)


class SRGAN(pl.LightningModule):
    def __init__(self, pretrain_gen: str = "", patch: bool = True, concat: bool = True):
        super(SRGAN, self).__init__()
        self.concat = concat
        self.patch = patch
        self.gpu: bool = torch.cuda.is_available()
        self.netG: nn.Module = SRResNet()
        self.netD: nn.Module = Discriminator(patch=patch, concat=concat)
        if not pretrain_gen:
            print("""Warning: Pretrain generator checkpoint not given!
        Training this model from scratch may lead to unwanted results!
        If you're resuming training from a checkpoint or performing evaluation on a trained model, ignore this warning!""")
        else:
            self.netG.load_from_checkpoint(pretrain_gen)

        if self.gpu:
            self.netD = self.netD.cuda()
            self.netG = self.netG.cuda()

        self.generated_imgs: Tensor = torch.empty(3, 128, 128)

    def forward(self, z):
        return self.netG(z)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch: List[Tensor], batch_nb, optimizer_idx: int):
        lr, hr, interpolated_lr = batch
        if self.gpu:
            lr, hr, interpolated_lr = lr.cuda(), hr.cuda(), interpolated_lr.cuda()
        self.generated_imgs = self(lr)

        if optimizer_idx == 1:
            real = torch.ones((hr.size(0), 1, 5, 5), device=self.device)

            d_fake: Tensor = self.netD(self.generated_imgs, interpolated_lr)

            g_loss = 0.001 * \
                self.adversarial_loss(d_fake, real) + \
                content_loss(self.generated_imgs, hr)

            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        elif optimizer_idx == 0:
            print("Disc train")
            real = 0.3 * torch.rand((hr.size(0), 1, 5, 5),
                                    device=self.device) + 0.7
            fake = 0.3 * torch.rand((hr.size(0), 1, 5, 5), device=self.device)
            # label smoothing, between 0.7-1.0 for real and 0.0 to 1.2 for fake

            real_loss = self.adversarial_loss(
                self.netD(hr, interpolated_lr), real)
            fake_loss = self.adversarial_loss(
                self.netD(self.generated_imgs.detach(), interpolated_lr), fake)

            d_loss = (fake_loss + real_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def test_step(self, batch: List[Tensor], batch_idx: int):
        lr, _, _ = batch
        sr = self(lr)
        with open(f'preds/{batch_idx}.png', 'wb+') as f:
            save_image(sr, f)
        pass

    def validation_step(self, batch: List[Tensor], batch_idx):
        lr, hr, interpolated_lr = batch
        if self.gpu:
            lr, hr, interpolated_lr = lr.cuda(), hr.cuda(), interpolated_lr.cuda()
        sr = self(lr)
        d_fake = self.netD(sr, interpolated_lr)
        real = torch.ones((hr.size(0), 1, 5, 5), device=self.device)

        val_loss = self.adversarial_loss(d_fake, real) + content_loss(sr, hr)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def configure_optimizers(self):
        opt_g = Adam(self.netG.parameters(), lr=0.0002,
                     betas=(0.5, 0.999), eps=1e-8)
        opt_d = Adam(self.netD.parameters(), lr=0.0002,
                     betas=(0.5, 0.999), eps=1e-8)
        # second array is for lr schedulers if needed
        return [opt_d, opt_g], []
