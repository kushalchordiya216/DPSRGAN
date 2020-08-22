from data import Dataset
from models import Generator, ContentLoss
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

Generator.compile(loss=[mean_squared_error, ContentLoss], optimizer=Adam(learning_rate=0.002, beta_1=0.5, epsilon=1e-8))
trainSet = Dataset(img_dir="./images", batch_size=512, max_iter=0)

Generator.fit(trainSet, epochs=30, steps_per_epoch=10)
