import torch
import torch.nn as nn

from torch import Tensor
from torchvision import models


class GeneratorHead(nn.Module):
    def __init__(self):
        super(GeneratorHead, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64,
                              kernel_size=9, stride=1, padding='same')
        self.PReLU = nn.PReLU(64)

    def forward(self, inp: Tensor):
        return self.PReLU(self.conv(inp))


class ResBlock(nn.Module):
    def __init__(self, dilation: bool = False):
        super(ResBlock, self).__init__()
        if dilation:
            self.conv1 = nn.Conv2d(in_channels=64, out_channels=64,
                                   kernel_size=3, stride=1, padding='same', dilation=2)
        else:
            self.conv1 = nn.Conv2d(in_channels=64, out_channels=64,
                                   kernel_size=3, stride=1, padding='same')

        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding='same')
        self.bn3 = nn.BatchNorm2d(num_features=64)

        self.PReLU1 = nn.PReLU(64)
        self.PReLU2 = nn.PReLU(64)

    def forward(self, inp: Tensor):
        X: Tensor = self.PReLU1(self.bn1(self.conv1(inp)))
        X = self.PReLU2(self.bn2(self.conv2(X)))
        X = self.bn3(self.conv3(X))
        return X.add(inp)  # Skip connections


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(64)
        )

    def forward(self, inp: Tensor):
        return self.main(inp)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.head = GeneratorHead()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.RRDB = nn.Sequential(
            *[ResBlock(dilation=True) if i % 2 == 0 else ResBlock() for i in range(16)])
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4)
        self.tail = nn.Sequential(UpSample(), UpSample(), self.conv2)

    def forward(self, inp: Tensor) -> Tensor:
        preRRDB: Tensor = self.head(inp)
        X: Tensor = self.RRDB(preRRDB)
        X = self.bn(self.conv1(X))
        X = X.add(preRRDB)  # skip conn
        X = self.tail(X)
        return X


# ############# Discriminator ##############


class DiscriminatorHead(nn.Module):
    def __init__(self, concat: bool = True):
        super(DiscriminatorHead, self).__init__()
        self.concat = concat
        self.conv = nn.Conv2d(
            in_channels=6, out_channels=64, kernel_size=3, stride=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, inp: Tensor, target: Tensor = torch.empty(1, 1)):
        if self.concat:
            inp: Tensor = torch.cat((inp, target), 1)
        return self.lrelu(self.conv(inp))


class DiscriminatorConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super(DiscriminatorConvBlock, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=3,
                                            stride=stride),
                                  nn.BatchNorm2d(num_features=out_channels),
                                  nn.LeakyReLU(negative_slope=0.2))

    def forward(self, inp: Tensor):
        return self.main(inp)


class Flatten(nn.Module):
    def forward(self, x: Tensor):
        return x.view(x.size(0), -1)


class DiscriminatorTail(nn.Module):
    def __init__(self, patch: bool = True):
        super(DiscriminatorTail, self).__init__()
        if not patch:
            print(
                "\033[93m If you choose to have a non-patch discriminator, "
                "make sure the discriminator architecture is on accordance with your image size\033[0m"
            )
            self.main = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=12800, out_features=1024),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(in_features=1024, out_features=1),
                nn.LeakyReLU(negative_slope=0.2)
            )
        else:
            self.main = nn.Sequential(nn.Conv2d(in_channels=512,
                                                out_channels=1, kernel_size=3,
                                                stride=1, padding=1))

    def forward(self, inp: Tensor):
        return torch.sigmoid(self.main(inp))


class Discriminator(nn.Module):
    def __init__(self, patch: bool = True, concat: bool = True):
        super(Discriminator, self).__init__()
        self.head = DiscriminatorHead(concat=concat)
        self.body = nn.Sequential(DiscriminatorConvBlock(64, 64, 2),
                                  DiscriminatorConvBlock(64, 128, 1),
                                  DiscriminatorConvBlock(128, 128, 2),
                                  DiscriminatorConvBlock(128, 256, 1),
                                  DiscriminatorConvBlock(256, 256, 2),
                                  DiscriminatorConvBlock(256, 512, 1),
                                  DiscriminatorConvBlock(512, 512, 2),
                                  )
        self.tail = DiscriminatorTail(patch=True)

    def forward(self, inp: Tensor, target: Tensor) -> Tensor:
        x: Tensor = self.head(inp, target)
        x = self.body(x)
        x = self.tail(x)
        return x


# ######## Perceptual Net ###########


class PerceptionNet(nn.Module):
    def __init__(self):
        super(PerceptionNet, self).__init__()
        modules = list(models.vgg19(pretrained=True).children())[0]
        self.main = nn.Sequential(*modules)
        for param in self.main.parameters():
            param.requires_grad = False

    def forward(self, inp):
        return self.main(inp)
