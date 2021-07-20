import argparse
import os
import sys

from pytorch_lightning import Trainer

from src.models import SRGAN, SRResNet
from utils.dataloader import SRDataLoader

parser = argparse.ArgumentParser(prog="Testing script",
                                 description="A script for testing out trained models on a set of test images")
parser.add_argument("model_path", type=str,
                    help='path to the pretrained model checkpoint file')
parser.add_argument("data_dir", type=str,
                    help='path to directory where images are stored, partitioned into train and test sub directories')
parser.add_argument("network", type=str, choices=["SRGAN", "SRResNet"], default="SRGAN",
                    help="type of network, either GAN or SRResNet")

args = parser.parse_args()

if __name__ == "__main__":
    if not args.model_path:
        print("Model path needs to be specified!")
        sys.exit(1)
    try:
        os.mkdir("preds")
    except FileExistsError:
        pass
    if args.network == "SRGAN":
        model = SRGAN.load_from_checkpoint(args.model_path)
    else:
        model = SRResNet.load_from_checkpoint(args.model_path)
    data = SRDataLoader(data_dir=args.data_dir, batch_size=1)
    data.setup('test')
    trainer: Trainer = Trainer(gpus=1)
    trainer.test(model, datamodule=data)
