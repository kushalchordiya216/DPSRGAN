import argparse
import os

from PIL import Image
from tqdm import tqdm

from utils.metrics import psnr, ssim

parser = argparse.ArgumentParser(
    prog="evaluation script",
    description="Given the path to generated images and ground truth, "
                "this script outputs the evaluation metrics for the images")

parser.add_argument('preds', type=str, default='preds/',
                    help='path to the directory where generated images are stored')
parser.add_argument('targets', type=str, default='images/',
                    help='path to the directory where ground truth or high resolution images are stored')

args = parser.parse_args()

if __name__ == "__main__":
    pred_dir = args.preds
    targets_dir = args.targets
    pred_names = os.listdir(pred_dir)
    targets_names = os.listdir(targets_dir)
    if pred_names != targets_names:
        print("""There are inconsistent filenames in the preds and targets directories
        The script assumes both sets have the same name for corrosponding images
        Please remove any extra or erroneous files in either set, make sure that the same encoding is used in both sets of images, etc.""")
    else:
        preds = [Image.open(os.path.join(os.getcwd(), name))
                 for name in pred_names]
        targets = [Image.open(os.path.join(os.getcwd(), name))
                   for name in targets_names]
        psnr_score = sum([psnr(pred, target)
                          for pred, target in tqdm(zip(preds, targets))]) / len(preds)
        print(f"PSNR score: {psnr_score}")
        ssim_score = sum([ssim(pred, target)
                          for pred, target in tqdm(zip(preds, targets))]) / len(preds)
        print(f"SSIM score: {ssim_score}")
