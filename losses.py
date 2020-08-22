import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from networks import PerceptionNet
from torchvision.transforms import Resize
from PIL.Image import BICUBIC


class ContentLoss(nn.Module):
    """Takes in the generated image and the target image, and passes both (separately) through a frozen VGG graph
    Then the output of the VGG activation layers is then used to calculate the Content loss by taking pizel-wise MSE


    Parameters
    ----------
        pred [Tensor] : generated prediction by the generator network
        target [Tensor]: Target or ground truth image for given prediction
    """

    def __init__(self):
        super(ContentLoss, self).__init__()
        self.VGG = PerceptionNet()

    def forward(self, pred: Tensor, target: Tensor):
        real, fake = self.VGG(target), self.VGG(pred)
        return F.mse_loss(fake, real)


class DownScaleLoss(nn.Module):
    """
    Takes the generatd image and the input image, downscales(resizes) the generated images to the dimensions
    of input image and calculates downscale loss by taking pixel-wise MSE loss
    Parameters
    ----------
        pred [Tensor] : generated prediction by the generator network
        target [Tensor]: input image for that prediction
    """

    def __init__(self, size: int = 32):
        super(DownScaleLoss, self).__init__()
        self.resize = Resize(size=(size, size), interpolation=BICUBIC)

    def forward(self, pred: Tensor, target: Tensor):
        downscaled = self.resize(pred)
        return F.mse_loss(downscaled, target)
