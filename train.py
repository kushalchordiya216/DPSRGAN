import os
from collections import OrderedDict

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from models import Generator, Discriminator, PerceptionNet
from dataloader import SRDataLoader


class SRGAN(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(SRGAN, self).__init__(*args, **kwargs)
        self.ngpu = 1
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")

        self.netG = Generator().to(self.device)
        self.netD = Discriminator().to(self.device)
        self.perceptual = PerceptionNet().to(self.device)

        self.generated_imgs = None
        self.last_imgs = None

    def forward(self, z):
        return self.netG(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def content_loss(self, fake, real):
        return F.mse_loss(self.perceptual(fake), self.perceptual(real))

    def training_step(self, batch, batch_nb, optimizer_idx):
        lr: Tensor = batch[0]
        hr: Tensor = batch[1]
        lr = lr.to(device)
        self.last_imgs: Tensor = hr.to(self.device)
        if optimizer_idx == 0:
            self.generated_imgs: Tensor = self(lr)
            valid = torch.ones(hr.size(0), 1, device=self.device)
            with torch.no_grad():
                D_fake: Tensor = self.netD(self.generated_imgs)
                real_activations = self.perceptual(self.hr)
                fake_activations = self.perceptual(self.lr)
            g_loss = self.adversarial_loss(D_fake, valid) + \
                self.content_loss(fake_activations, real_activations)
            tqdm_dict = {'g_loss': g_loss}
            return OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

        elif optimizer_idx == 1:
            valid = torch.ones(hr.size(0), 1, device=self.device)
            fake = torch.zeros(hr.size(0), 1, device=self.device)

            real_loss = self.adversarial_loss(self.netD(hr), valid)
            fake_loss = self.adversarial_loss(self.netD(self.netG(lr).detach()), fake)
            d_loss = (fake_loss + real_loss)/2
            tqdm_dict = {'d_loss': d_loss}
            return OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

    def configure_optimizers(self):
        super().configure_optimizers()
        optG = Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-8)
        optD = Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-8)
        return [optG, optD], []

    def on_epoch_end(self):
        # log sampled images
        for lr, hr in val_loader:
            lr = lr.to(self.device)
        sample_imgs = self(lr)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)


gan_model = GAN(hparams)
data = SRDataLoader()
# most basic trainer, uses good defaults (1 gpu)
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(os.getcwd(), 'models'),
    save_top_k=1,
    verbose=True,
    monitor='g_loss',
    mode='min',
    prefix=''
)

trainer = pl.Trainer(gpus=1, checkpoint_callback=checkpoint_callback)
trainer.fit(gan_model, datamodule=data)
