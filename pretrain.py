import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from models import Generator, PerceptionNet
import torch.nn.functional as F
from torch.optim import Adam
from dataloader import SRDataLoader


class PreTrainGenModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = Generator()
        self.VGG = PerceptionNet()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        hr_activations, sr_activations = self.VGG(hr), self.VGG(sr)
        loss = F.mse_loss(sr, hr) + F.mse_loss(sr_activations, hr_activations)

        self.logger.summary.scalar('loss', loss, step=self.global_step)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return result

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr) + F.mse_loss(sr_activations, hr_activations)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': test_loss_mean}

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.0002)


data = SRDataLoader()

model = PreTrainGenModel()
checkpoint_model = ModelCheckpoint(filepath="./models/", verbose=True, prefix="pretraingen")
trainer = pl.Trainer(checkpoint_callback=checkpoint_model, gpus=1, max_epochs=150)

trainer.fit(model, data)
trainer.test(datamodule=data)
