#!/usr/bin/env python
# coding: utf-8

# In[7]:
# In[20]:


import os
from collections import OrderedDict
from glob import glob
from argparse import Namespace
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import save_image
from torchvision import models
from torchvision.transforms import Resize, ToTensor

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer


# In[21]:


class GeneratorHead(nn.Module):
    def __init__(self):
        super(GeneratorHead, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4)
        self.PReLU = nn.PReLU(64)

    def forward(self, inp: Tensor):
        return self.PReLU(self.conv(inp))


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.PReLU = nn.PReLU(64)

    def forward(self, inp: Tensor):
        X: Tensor = self.PReLU(self.bn(self.conv(inp)))
        X: Tensor = self.bn(self.conv(X))
        return X.add(inp)  # Skip connections


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(64)
        )

    def forward(self, inp: Tensor):
        return self.main(inp)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.head = GeneratorHead()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.RRDB = nn.Sequential(*[ResBlock() for _ in range(16)])
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4)
        self.tail = nn.Sequential(UpSample(), UpSample(), self.conv2)

    def forward(self, inp: Tensor) -> Tensor:
        preRRDB: Tensor = self.head(inp)
        X: Tensor = self.RRDB(preRRDB)
        X: Tensor = self.bn(self.conv1(X))
        X: Tensor = X.add(preRRDB)  # skip conn
        X: Tensor = self.tail(X)
        return X

# ################################################ Discriminator ############################################


class DiscriminatorHead(nn.Module):
    def __init__(self):
        super(DiscriminatorHead, self).__init__()
        self.conv = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, inp: Tensor, target: Tensor):
        concat: Tensor = torch.cat((inp, target), 1)
        return self.leaky_relu(self.conv(concat))


class DiscriminatorConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super(DiscriminatorConvBlock, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                            stride=stride),
                                  nn.BatchNorm2d(num_features=out_channels),
                                  nn.LeakyReLU(negative_slope=0.2))

    def forward(self, inp: Tensor):
        return self.main(inp)


class Flatten(nn.Module):
    def forward(self, X: Tensor):
        return X.view(X.size(0), -1)


class DiscriminatorTail(nn.Module):
    def __init__(self, patch: bool = True):
        super(DiscriminatorTail, self).__init__()
        if not patch:
            self.main = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=12800, out_features=1024),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(in_features=1024, out_features=1),
                nn.LeakyReLU(negative_slope=0.2)
            )
        else:
            self.main = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3,
                                                stride=1, padding=1))

    def forward(self, inp: Tensor):
        return torch.sigmoid(self.main(inp))


class Discriminator(nn.Module):
    def __init__(self, patch: bool = True):
        super(Discriminator, self).__init__()
        self.head = DiscriminatorHead()
        self.body = nn.Sequential(DiscriminatorConvBlock(64, 64, 2),
                                  DiscriminatorConvBlock(64, 128, 1),
                                  DiscriminatorConvBlock(128, 128, 2),
                                  DiscriminatorConvBlock(128, 256, 1),
                                  DiscriminatorConvBlock(256, 256, 2),
                                  DiscriminatorConvBlock(256, 512, 1),
                                  DiscriminatorConvBlock(512, 512, 2),
                                  )
        self.tail = DiscriminatorTail(patch=patch)

    def forward(self, inp: Tensor, target: Tensor) -> Tensor:
        x: Tensor = self.head(inp, target)
        x: Tensor = self.body(x)
        x: Tensor = self.tail(x)
        return x

# ################################################# Perceptual Net ###########################################


class PerceptionNet(nn.Module):
    def __init__(self):
        super(PerceptionNet, self).__init__()
        modules = list(models.vgg19(pretrained=True).children())[0]
        self.main = nn.Sequential(*modules)
        for param in self.main.parameters():
            param.requires_grad = False

    def forward(self, inp):
        return self.main(inp)


# In[33]:


class ContentLoss(nn.Module):
    def __init__(self):
        """Takes in the generated image and the target image, and passes both (separately) through a frozen VGG graph
            Then the output of the VGG activation layers is then used to calculate the Content loss by taking
            pixel-wise MSE


            Args
            ----------
                pred [Tensor] : generated prediction by the generator network
                target [Tensor]: Target or ground truth image for given prediction
        """
        super(ContentLoss, self).__init__()
        self.VGG = PerceptionNet().cuda()

    def forward(self, pred: Tensor, target: Tensor):
        real, fake = self.VGG(target), self.VGG(pred)
        return F.mse_loss(fake, real)


class DownScaleLoss(nn.Module):
    def __init__(self, size: int = 32):
        """
            Takes the generated image and the input image, downscales(resizes) the generated images to the dimensions
            of input image and calculates downscale loss by taking pixel-wise MSE loss

            Args
            ----------
                pred [Tensor] : generated prediction by the generator network
                target [Tensor]: input image for that prediction
        """
        super(DownScaleLoss, self).__init__()
        self.size = size
        self.VGG = PerceptionNet().cuda()

    def forward(self, pred: Tensor, interpolated: Tensor):
        pred, interpolated = self.VGG(pred), self.VGG(interpolated)
        return F.mse_loss(pred, interpolated)


# In[38]:


def recursiveResize(img: Image, factor: int = 2):
    """
    Recursively resizes an image by down scaling by 2,
    repeats this for factor times
    Args:
        img (PIL.Image): image to be resized
        factor (int): factor by which resizing is to take place. eg. if factor is 2, image will be downscaled
        to half it's size twice, thereby final image with be 1/4th the original image size
    Returns:

    """
    for _ in range(factor):
        height, width = img.size
        resize = Resize((int(height / 2), int(width / 2)), interpolation=Image.BICUBIC)
        img = resize(img)
    return img


class SRDataset(Dataset):
    def __init__(self, img_dir: str = './images/*.jpg'):
        super(SRDataset, self).__init__()
        self.img_dir = img_dir
        self.filenames = glob(self.img_dir)
        self.toTensor = ToTensor()
        self.setSize = Resize((128, 128), interpolation=Image.BICUBIC)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img = Image.open(self.filenames[index])
        img = self.setSize(img)
        lr_img = recursiveResize(img, 2)
        hr_img = img
        interpolated_img = self.setSize(lr_img)
        lr_img, hr_img, interpolated_img = self.toTensor(lr_img), self.toTensor(hr_img), self.toTensor(interpolated_img)
        return lr_img, hr_img, interpolated_img


class SRDataLoader(LightningDataModule):
    def __init__(self, data_dir: str = "./images/*.jpg", batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.train, self.val, self.test = None, None, None

    def setup(self, stage=None):
        print(len(SRDataset(img_dir=self.data_dir)))
        self.train, self.val, self.test = random_split(
            SRDataset(img_dir=self.data_dir), lengths=[202535, 32, 32], generator=torch.Generator().manual_seed(69))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4, drop_last=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4, pin_memory=True)


# In[35]:


data = SRDataLoader(data_dir='/datasets/celebA/*.jpg')
data.setup()


class LogImages(Callback):
    def on_epoch_end(self, model: Trainer, pl_module: LightningModule):
        for lr_input, _, _ in data.val_dataloader():
            lr_input = lr_input.cuda()
            preds = pl_module.netG(lr_input)
            with open(f'preds{model.current_epoch}.png', 'wb+') as file:
                save_image(preds, file, nrow=8)


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

    def on_epoch_end(self, model: Trainer, pl_module):
        if model.current_epoch % self.run_every_e == 0:
            torch.save(pl_module.netG, os.path.join(self.save_dir, f"generator_{model.current_epoch}.ckpt"))
            if model.current_epoch >= self.save_last_k:
                os.remove(os.path.join(self.save_dir, f"generator_{model.current_epoch - self.save_last_k}.ckpt"))


# In[42]:


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
        lr_input, hr_input, _ = batch
        lr_input, hr_input = lr_input.cuda(), hr_input.cuda()
        sr = self(lr_input)
        loss = F.mse_loss(sr, hr_input) + content_loss(sr, hr_input)

        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return result

    def validation_step(self, batch, batch_idx):
        lr_input, hr_input, _ = batch
        lr_input, hr_input = lr_input.cuda(), hr_input.cuda()
        sr = self(lr_input)
        loss = F.mse_loss(sr, hr_input) + content_loss(sr, hr_input)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def configure_optimizers(self):
        return Adam(self.netG.parameters(), lr=0.0002)


class SRGAN(pl.LightningModule):
    def __init__(self, hyper_params):
        super(SRGAN, self).__init__()
        self.hparams = hyper_params

        self.netG: nn.Module = Generator().cuda()
        self.netD: nn.Module = Discriminator().cuda()

        self.generated_imgs = None
        self.last_imgs = None

    def forward(self, z):
        return self.netG(z)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb, optimizer_idx):
        lr_input, hr_input, interpolated_lr = batch
        lr_input, hr_input, interpolated_lr = lr_input.cuda(), hr_input.cuda(), interpolated_lr.cuda()
        self.last_imgs: Tensor = hr_input
        self.generated_imgs: Tensor = self(lr_input)
        if optimizer_idx == 1:
            real = torch.ones((hr_input.size(0), 1, 5, 5), device=self.device)

            # with torch.no_grad():
            D_fake: Tensor = self.netD(self.generated_imgs, interpolated_lr)

            g_loss = 0.001 * self.adversarial_loss(D_fake, real) + content_loss(self.generated_imgs, hr_input)

            if self.hparams.downscale_loss:
                g_loss += downscale_loss(self.generated_imgs, interpolated_lr)

            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        elif optimizer_idx == 0:
            real = 0.3 * torch.rand((hr_input.size(0), 1, 5, 5), device=self.device) + 0.7
            fake = 0.3 * torch.rand((hr_input.size(0), 1, 5, 5), device=self.device)
            # label smoothing, between 0.7-1.0 for real and 0.0 to 1.2 for fake

            real_loss = self.adversarial_loss(self.netD(hr_input, interpolated_lr), real)
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
        lr_input, hr_input, interpolated_lr = batch
        lr_input, hr_input, interpolated_lr = lr_input.cuda(), hr_input.cuda(), interpolated_lr.cuda()
        sr = self(lr_input)
        D_fake = self.netD(sr, interpolated_lr)
        real = torch.ones((hr_input.size(0), 1, 5, 5), device=self.device)

        val_loss = self.adversarial_loss(D_fake, real) + content_loss(sr, hr_input)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def configure_optimizers(self):
        opt_g = Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-8)
        opt_d = Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-8)
        return [opt_d, opt_g], []  # second array is for lr schedulers if needed


# In[43]:


args = {
    'batch_size': 32,
    'lr': 0.0002,
    'b1': 0.5,
    'b2': 0.999,
    'data_dir': '/datasets/celebA/*.jpg',
    'epochs': 15,
    'downscale_loss': False,
    'model_save_dir': '/storage/'
}
hparams = Namespace(**args)

data = SRDataLoader(data_dir=hparams.data_dir, batch_size=hparams.batch_size)
data.setup('fit')

gan_model = SRGAN(hparams)
checkpoint_callback = ModelCheckpoint(
    filepath=hparams.model_save_dir,
    save_top_k=1,
    verbose=True,
    monitor='g_loss',
    mode='min',
    prefix='concat_downscale_perceptual',
    save_last=True
)
#
# checkpoint = torch.load(os.path.join(hparams.model_save_dir, 'pretrain_gen.ckpt'))
# temp = {}
# for key, value in checkpoint['state_dict'].items():
#     if 'net' in key:
#         key = key.replace('net', 'netG')
#         temp[key] = value
# checkpoint['state_dict'] = temp
# gan_model.load_state_dict(checkpoint['state_dict'], strict=False)

for lr, hr, interpolate_hr in data.val_dataloader():
    with open('input.png', 'wb+') as f:
        save_image(lr, f)
    with open('ground_truth.png', 'wb+') as f:
        save_image(hr, f)
    with open('interpolated.png', 'wb+') as f:
        save_image(interpolate_hr, f)

trainer = pl.Trainer(checkpoint_callback=checkpoint_callback, max_epochs=hparams.epochs,
                     callbacks=[LogImages(), CustomCheckpoint(save_dir=hparams.model_save_dir, save_last_k=2)], gpus=1)
trainer.fit(gan_model, datamodule=data)
# trainer.test(gan_model, datamodule=data)


# In[ ]:


# In[ ]:
