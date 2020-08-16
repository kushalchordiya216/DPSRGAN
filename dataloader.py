from torch.utils.data import Dataset, DataLoader, random_split
from glob import glob
from PIL import Image
import torch
from torchvision.transforms import Resize, ToTensor

from pytorch_lightning import LightningDataModule


class SRDataset(Dataset):
    def __init__(self, img_dir: str = './images/*.jpg'):
        super(SRDataset, self).__init__()
        self.img_dir = img_dir
        self.filenames = glob(self.img_dir)
        self.toTensor = ToTensor()

    def __len__(self):
        return len(self.filenames)

    def recursiveResize(self, img, size: int):
        sizes = [128, 64, 32, 16]
        for s in sizes:
            resize = Resize((s, s), interpolation=Image.BICUBIC)
            img = resize(img)
            if s == size:
                break
        return img

    def __getitem__(self, index):
        img = Image.open(self.filenames[index])
        lr_img = self.toTensor(self.recursiveResize(img, 32))
        hr_img = self.toTensor(self.recursiveResize(img, 128))
        return lr_img, hr_img


class SRDataLoader(LightningDataModule):
    def __init__(self, data_dir: str = "./images/*.jpg", batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train, self.val, self.test = random_split(
            SRDataset(), lengths=[128000, 32, 74567], generator=torch.Generator().manual_seed(42))

    def train_dataloader(self, *args, **kwargs):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self, *args, **kwargs):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self, *args, **kwargs):
        return DataLoader(self.test, batch_size=self.batch_size)
