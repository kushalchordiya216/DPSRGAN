import os
from argparse import Namespace
from collections import OrderedDict
from PIL import Image
from torchvision.transforms import ToPILImage,  ToTensor
from torchvision.utils import save_image

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloader import SRDataLoader, recursiveResize
from models import SRGAN, PreTrainGenModel
from callbacks import LogImages

args = {
    'batch_size': 1,
    'lr': 0.0002,
    'b1': 0.5,
    'b2': 0.999,
    'data_dir': './images/*.jpg',
}
hparams = Namespace(**args)

data = SRDataLoader(data_dir=hparams.data_dir, batch_size=hparams.batch_size)
data.setup('fit')

gan_model = SRGAN(hparams)
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    verbose=True,
    monitor='g_loss',
    mode='min',
    prefix='concat'
)

checkpoint = torch.load('./models/pretrain_gen.ckpt', map_location=torch.device('cpu'))
temp = {}
for key, value in checkpoint['state_dict'].items():
    if 'net' in key:
        key = key.replace('net', 'netG')
        temp[key] = value
checkpoint['state_dict'] = temp
print(checkpoint['state_dict'].keys())
gan_model.load_state_dict(checkpoint['state_dict'], strict=False)

trainer = pl.Trainer(checkpoint_callback=checkpoint_callback, max_epochs=15, callbacks=[LogImages()])
trainer.fit(gan_model, datamodule=data)
trainer.test(model, datamodule=data)
