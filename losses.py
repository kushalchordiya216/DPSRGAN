import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from networks import PerceptionNet
from torchvision.transforms import Resize
from PIL.Image import BICUBIC


class ContentLoss(nn.Module):
    def __init__(self):
        """Takes in the generated image and the target image, and passes both (separately) through a frozen VGG graph
            Then the output of the VGG activation layers is then used to calculate the Content loss by taking pizel-wise MSE


            Args
            ----------
                pred [Tensor] : generated prediction by the generator network
                target [Tensor]: Target or ground truth image for given prediction
        """
        super(ContentLoss, self).__init__()
        self.VGG = PerceptionNet()

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

    def forward(self, pred: Tensor, target: Tensor):
        downscaled = F.interpolate(pred, (self.size, self.size))
        return F.mse_loss(downscaled, target)
