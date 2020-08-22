import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloader import SRDataLoader
from models import PreTrainGenModel
from callbacks import LogImages

data = SRDataLoader(batch_size=32)
data.setup()

model = PreTrainGenModel()
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    verbose=True,
)

trainer = pl.Trainer(checkpoint_callback=checkpoint_callback, max_epochs=15, callbacks=[LogImages()])
trainer.fit(model, datamodule=data)
trainer.test(model, datamodule=data)
