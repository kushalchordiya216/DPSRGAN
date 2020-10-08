import numpy as np
from skimage.measure import compare_ssim as SSIM


def psnr(pred: np.ndarray, target: np.ndarray):
    mse: np.float64 = np.mean(np.square(target - pred))
    max_f = 255
    return 10 * np.log10(max_f / mse)


def ssim(pred: np.ndarray, target: np.ndarray):
    return SSIM(target, pred)
