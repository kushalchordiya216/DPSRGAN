from argparse import Namespace

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.dataloader import SRDataLoader
from src.models import SRGAN

args = {
    'batch_size': 1,
    'lr': 0.0002,
    'b1': 0.5,
    'b2': 0.999,
    'data_dir': './images/*.jpg',
    'epochs': 15,
    'downscale_loss': True,
    'model_save_dir': './models/'
}
hparams = Namespace(**args)


gan_model = SRGAN()
checkpoint_callback = ModelCheckpoint(
    filepath=hparams.model_save_dir,
    save_top_k=2,
    verbose=True,
    monitor='g_loss',
    mode='min',
    prefix='concat_downscale',
    save_last=True
)

data = SRDataLoader(data_dir=hparams.data_dir)
data.setup('fit')
