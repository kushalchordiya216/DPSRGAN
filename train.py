import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.models import SRResNet, SRGAN
from utils.dataloader import SRDataLoader
from utils.callbacks import LogImages

parser = argparse.ArgumentParser(prog="Training script",
                                 description="Script for training models on different architectures")

parser.add_argument("batch_size", type=int, default=32,
                    help="number of images to be passed to the network for training one mini_batch")
parser.add_argument('epochs', type=int, default=50,
                    help="number of epochs for which model is to be trained")
parser.add_argument('learning_rate', type=float, default=0.0002,
                    help="learning rate for adam optimizer")
parser.add_argument('beta_1', type=float, default=0.5,
                    help="beta_1 value for Adam optimizer")
parser.add_argument('beta_2', type=float, default=0.999,
                    help="beta_2 value for Adam optimizer")
parser.add_argument('data_dir', type=str, default='images/',
                    help="relative path of the directory having HR images on which models are to be trained on")

parser.add_argument('network', type=str, default='SRResNet', choices=['SRGAN', 'SRResNet'],
                    help='type of network, whether to train SRGAN or SRResNet')
parser.add_argument('patch', type=bool, default=True,
                    help="specify whether to use a patch discriminator")
parser.add_argument('concat', type=bool, default=True,
                    help="specify whether to feed a concatenated input to discriminator")
parser.add_argument('pretrain_gen', type=str,
                    help="name of pretrained generator to be used for training SRGAN")
parser.add_argument('checkpoint', type=str, default=None,
                    help='checkpoint, if any, to be used to restart training')
parser.add_argument('model_dir', type=str, default="models/",
                    help="directory where trained models are to be saved")
parser.add_argument('save_best', type=int, default=2,
                    help='number of k best models to save')

args = parser.parse_args()

if __name__ == '__main__':
    if args.network == 'SRGAN':
        model = SRGAN(args.pretrain_gen, args.patch, args.concat)
    else:
        model = SRResNet()
    data = SRDataLoader(data_dir=args.data_dir, batch_size=args.batch_size)
    data.setup('fit')

    checkpoint_callback = ModelCheckpoint(
        filepath=args.model_dir,
        save_top_k=args.save_best,
        verbose=True,
        monitor='g_loss',
        mode='min',
        save_last=True,
    )

    trainer: Trainer = Trainer(max_epochs=args.epochs, checkpoint_callback=checkpoint_callback, callbacks=[LogImages()],
                               gpus=1)
    if args.checkpoint:
        trainer: Trainer = Trainer(max_epochs=args.epochs, checkpoint_callback=checkpoint_callback,
                                   callbacks=[LogImages()], gpus=1, resume_from_checkpoint=args.checkpoint)

    trainer.fit(model=model, datamodule=data)
