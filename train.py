from argparse import Namespace
from torchvision.utils import save_image

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloader import SRDataLoader
from models import SRGAN
from callbacks import LogImages, CustomCheckpoint

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

data = SRDataLoader(data_dir=hparams.data_dir, batch_size=hparams.batch_size)
data.setup('fit')

gan_model = SRGAN(hparams)
checkpoint_callback = ModelCheckpoint(
    filepath=hparams.model_save_dir,
    save_top_k=2,
    verbose=True,
    monitor='g_loss',
    mode='min',
    prefix='concat_downscale',
    save_last=True
)

checkpoint = torch.load('./models/pretrain_gen.ckpt', map_location=torch.device('cpu'))
temp = {}
for key, value in checkpoint['state_dict'].items():
    if 'net' in key:
        key = key.replace('net', 'netG')
        temp[key] = value
checkpoint['state_dict'] = temp
gan_model.load_state_dict(checkpoint['state_dict'], strict=False)

for lr, hr, interpolate_hr in data.val_dataloader():
    with open('input.png', 'wb+') as f:
        save_image(lr, f)
    with open('ground_truth.png', 'wb+') as f:
        save_image(hr, f)
    with open('interpolated.png', 'wb+') as f:
        save_image(interpolate_hr, f)

trainer = pl.Trainer(checkpoint_callback=checkpoint_callback, max_epochs=hparams.epochs,
                     callbacks=[LogImages(), CustomCheckpoint(save_dir=hparams.model_save_dir)])
trainer.fit(gan_model, datamodule=data)
trainer.test(gan_model, datamodule=data)
