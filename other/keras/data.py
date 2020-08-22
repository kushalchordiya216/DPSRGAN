import cv2
import glob
import numpy as np
from keras.utils import Sequence
from math import ceil


def RecursiveResize(img, size):
    sizes = [256, 128, 64, 32]
    for i in sizes:
        img = cv2.resize(img, (i, i))
        if i == size:
            break
    return img


class Dataset(Sequence):
    def __getitem__(self, index):
        filenames = self.file_names[index*self.batch_size:(index + 1) * self.batch_size]
        print(filenames[0])
        return np.array([cv2.resize(cv2.imread(filename), (32, 32)) for filename in filenames[:1]]), np.array([
            cv2.resize(cv2.imread(filename), (64, 64)) for filename in filenames[:1]])

    def __len__(self):
        return ceil(self.max_iter / self.batch_size)

    def __init__(self, img_dir: str, batch_size: int = 32, max_iter: int = 0):
        self.file_names: list = sorted(glob.glob(f'{img_dir}/*.jpg'))
        self.index: int = 0
        if max_iter:
            self.max_iter: int = max_iter
        else:
            self.max_iter: int = len(self.file_names)
        self.batch_size: int = batch_size
