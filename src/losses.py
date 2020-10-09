from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from src.networks import PerceptionNet


class ContentLoss(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        raise NotImplementedError

    def __init__(self):
        """Takes in the generated image and the target image, and passes both (separately) through a frozen VGG graph
            Then the output of the VGG activation layers is then used to calculate the Content loss by taking pizel-wise MSE


            Args
            ----------
                pred [Tensor] : generated prediction by the generator network
                target [Tensor]: Target or ground truth image for given prediction
        """
        super(ContentLoss, self).__init__()
        self.gpu = torch.cuda.is_available()
        self.VGG = PerceptionNet()
        if self.gpu:
            self.VGG = self.VGG.cuda()

    def forward(self, pred: Tensor, target: Tensor):
        real, fake = self.VGG(target), self.VGG(pred)
        return F.mse_loss(fake, real)
