# %%
import tensorflow as tf
import tensorflow.python.keras.layers as L
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.optimizers import adam
import cv2
import os
import numpy as np

BATCH_SIZE = 256
EPOCH = 50
# %%


def ResBlock(x, filters):
    res = x
    x = L.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = L.ReLU()(x)
    x = L.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = L.ReLU()(x)
    x = L.Add()([x, res])
    return x


def create_generator() -> Model:
    img = L.Input(shape=(64, 64, 3))
    x = L.Conv2D(filters=8, kernel_size=(3, 3),
                 padding='same', activation='relu')(img)
    x = L.BatchNormalization()(x)
    x = ResBlock(x, 8)
    x = L.BatchNormalization()(x)
    x = ResBlock(x, 8)
    x = L.BatchNormalization()(x)
    x = ResBlock(x, 8)
    x = L.BatchNormalization()(x)
    x = L.Dropout(rate=0.2)(x)

    x = L.Conv2D(filters=16, kernel_size=(3, 3),
                 padding='same', activation='relu')(x)
    x = L.BatchNormalization()(x)
    x = ResBlock(x, 16)
    x = L.BatchNormalization()(x)
    x = ResBlock(x, 16)
    x = L.BatchNormalization()(x)
    x = ResBlock(x, 16)
    x = L.BatchNormalization()(x)
    x = L.Dropout(rate=0.2)(x)

    x = L.Conv2D(filters=32, kernel_size=(3, 3),
                 padding='same', activation='relu')(x)
    x = L.BatchNormalization()(x)
    x = ResBlock(x, 32)
    x = L.BatchNormalization()(x)
    x = ResBlock(x, 32)
    x = L.BatchNormalization()(x)
    x = ResBlock(x, 32)
    x = L.BatchNormalization()(x)
    x = L.Dropout(rate=0.2)(x)

    x = L.Conv2D(filters=3, kernel_size=(3, 3),
                 padding='same', activation='sigmoid')(x)
    gen = Model(inputs=img, outputs=x)
    gen.compile(loss=mean_squared_error,
                optimizer=adam(0.001))
    return gen
# %%


def load_inputs(path_x, path_y):
    x = []
    y = []
    while (True):
        for i in os.listdir(path_x):
            x.append(cv2.imread(path_x + i))
            y.append(cv2.imread(path_y + i))
            if len(x) == BATCH_SIZE:
                x = np.array(x)
                y = np.array(y)
                yield x, y
                x = []
                y = []


if __name__ == "__main__":
    generator: Model = create_generator()
    generator.fit_generator(
        load_inputs(str('data/celeba_sr/celeba_resized_inp/'),
                    str('data/celeba_sr/celeba_resized/')),
        steps_per_epoch=700, epochs=50,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath='pretrained_generator.hdf5', save_best_only=True)])


# %%
