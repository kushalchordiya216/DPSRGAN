from glob import glob
from PIL import Image
import torch
from torchvision.transforms import Resize, ToTensor
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule


def recursiveResize(img, factor: int):
    for _ in range(factor):
        height, width = img.size
        resize = Resize((height / 2, width / 2), interpolation=Image.BICUBIC)
        img = resize(img)
    return img


class SRDataset(Dataset):
    def __init__(self, img_dir: str = './images/*.jpg'):
        super(SRDataset, self).__init__()
        self.img_dir = img_dir
        self.filenames = glob(self.img_dir)
        self.toTensor = ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img = Image.open(self.filenames[index])
        lr_img = recursiveResize(img, 2)
        hr_img = recursiveResize(img, 2)
        interpolated_img = recursiveResize(lr_img, 128)
        lr_img, hr_img, interpolated_img = self.toTensor(lr_img), self.toTensor(hr_img), self.toTensor(interpolated_img)
        return lr_img, hr_img, interpolated_img


class SRDataLoader(LightningDataModule):
    def __init__(self, data_dir: str = "./images/*.jpg", batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.train, self.val, self.test = None, None, None

    def setup(self, stage=None):
        self.train, self.val, self.test = random_split(
            SRDataset(img_dir=self.data_dir), lengths=[4, 4, 202591], generator=torch.Generator().manual_seed(69))

    def train_dataloader(self, *args, **kwargs):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4, drop_last=True,
                          pin_memory=True)

    def val_dataloader(self, *args, **kwargs):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def test_dataloader(self, *args, **kwargs):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4, pin_memory=True)
