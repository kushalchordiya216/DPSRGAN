# %%
from datetime import datetime

import tensorflow as tf
from keras.layers import (Dense, Flatten, Input, Conv2D)
from keras.losses import binary_crossentropy, mean_squared_error
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.models import Model
from .utils import generator_block,discriminator_block,image_loader,build_perceptual,write_logs
from keras.backend import get_value

import os
import cv2
import sys
import numpy as np
from tqdm import tqdm
# %%
PARAMS = {'n_latent': 100, 'batch_size': 32, 'epochs': 50, 'steps_per_epoch': 200,
          'path_input': '/home/kushal/WorkSpace/Python/data/celeba_sr/celeba_resized_inp/',
          'path_output': '/home/kushal/WorkSpace/Python/data/celeba_sr/celeba_resized/',
          'disc_looping_factor': 2}

log_dir_gan = "./logs/gan/" + datetime.now().strftime("%Y%m%d")
log_dir_disc = "./logs/discriminator/" + datetime.now().strftime("%Y%m%d")


# %%


class SR_GAN(tf.keras.Model):
    def compute_output_signature(self, input_signature):
        pass

    def __init__(self, *args, **kwargs):
        super(SR_GAN, self).__init__()
        self.optimizer = Adam(lr=0.0001, epsilon=1e-08)

        print("building GAN ....")
        self.generator: Model = self.build_generator()
        self.discriminator: Model = self.build_discriminator()
        self.perceptual_model: Model = build_perceptual()
        self.GAN = self.build_GAN()

        print("Initializing tensorboard summaries")
        self.gan_callback = TensorBoard(log_dir=log_dir_gan, histogram_freq=1, batch_size=PARAMS['batch_size'],
                                        write_graph=True, write_grads=True)
        self.gan_callback.set_model(self.GAN)
        self.gan_metrics = ['total', 'cross_entropy', 'perceptual']
        self.disc_callback = TensorBoard(log_dir=log_dir_disc, histogram_freq=1, batch_size=PARAMS['batch_size'],
                                         write_graph=True, write_grads=True, write_images=True)
        self.disc_callback.set_model(self.discriminator)
        self.disc_metrics = ['cross_entropy']
        self.val_set: tuple = self.load_val_set()
        self.val_metrics = ['val_total', 'val_cross_entropy', 'val_perceptual']

    def perceptual_loss(self, images_true, images_pred):
        try:
            outputs_true = self.perceptual_model(images_true)
            outputs_pred = self.perceptual_model(images_pred)
            assert len(outputs_pred) == 4
            p_loss = 0
            for output_true, output_pred in zip(outputs_true, outputs_pred):
                p_loss += get_value(mean_squared_error(y_true=output_true, y_pred=output_pred))
            return 0.001 * p_loss
        except AssertionError as ae:
            print(ae, "Vggmodel outputs not as expected, please check")

    def build_perceptual(self)->Model:
        pass

    def build_generator(self) -> Model:
        """
        build Generator with RRDB blocks
        accepts input of shape (64,64,3), i.e. up sized images
        :rtype: tf.keras.model.Model
        """
        print("building Generator ....")
        input_image = Input(shape=(64, 64, 3))
        x = generator_block(input_image=input_image,filters=8)
        x = generator_block(input_image=x,filters=16)
        x = generator_block(input_image=x,filters=8)
        x = Conv2D(filters=3,kernel_size=(5,5), padding='same')(x)
        g_model = Model(input_image, x)
        g_model.name = "Generator"
        g_model.compile(loss=binary_crossentropy,
                        optimizer=self.optimizer)

        plot_model(g_model, "srgan/created_models/RRDB_Generator.png")
        print(g_model.summary())
        return g_model

    def build_discriminator(self) -> Model:
        """
        :rtype: tf.keras.models.Model
        """
        print("building Discriminator ....")
        inputs = Input(shape=(64, 64, 3))
        x = discriminator_block(inputs)
        x = discriminator_block(x)
        x = discriminator_block(x)
        x = discriminator_block(x)
        x = discriminator_block(x)
        x = discriminator_block(x)
        x = discriminator_block(x)
        x = Flatten()(x)
        x = Dense(1)(x)
        d_model = Model(inputs, x)
        d_model.name = 'Discriminator'
        d_model.compile(loss=binary_crossentropy, optimizer=self.optimizer)
        plot_model(d_model, "srgan/created_models/RRDB_discriminator.png")
        print(d_model.summary())
        return d_model

    def build_GAN(self) -> Model:
        print("building GAN ....")
        input_images = Input(shape=(64, 64, 3))
        x = self.generator(input_images)
        output = self.discriminator(x)
        perceptual_output = self.perceptual_model(x)
        gan = Model(inputs=input_images, outputs=[output, x])
        self.discriminator.trainable = False
        gan.compile(optimizer=self.optimizer, loss=[binary_crossentropy, self.perceptual_loss])
        gan.name = 'SRGAN'
        plot_model(gan, "srgan/created_models/RRDB_GAN.png")
        return gan

    def train_epoch(self, g_logs=0, d_loss=0):
        y_true = np.ones(PARAMS['batch_size']) - np.random.random_sample(
            PARAMS['batch_size']) * 0.2
        y_pred = np.zeros(PARAMS['batch_size']) + np.random.random_sample(
            PARAMS['batch_size']) * 0.2

        # ! y_true and y_pred being generated once per epoch, and the same being used across the epoch.
        # Not sure if this is a goood idea , but it will save time
        for batch_no in tqdm(range(PARAMS['steps_per_epoch'])):
            for i in range(PARAMS['disc_looping_factor']):
                # TODO: use proper generator instead of some hack workaround
                input_images, ground_truth = image_loader()
                pred_images = self.generator.predict(x=input_images)
                y_pred = np.reshape(y_pred, (y_pred.shape[0], 1))
                y_true = np.reshape(y_true, (y_true.shape[0], 1))

                disc_label = np.concatenate((y_true, y_pred), axis=0)
                disc_input = np.concatenate((input_images, pred_images), axis=0)
                d_loss += self.discriminator.train_on_batch(x=disc_input, y=disc_label)
            input_images, ground_truth = image_loader()
            g_logs += self.GAN.train_on_batch(x=input_images, y=[y_true, input_images])
            return g_logs, d_loss

    def load_val_set(self):
        try:
            val_set_addr = os.listdir(os.getcwd() + '../../data/celeba_sr/X/')
            val_set_X = []
            for i in val_set_addr:
                val_set_X.append(cv2.imread(os.getcwd() + '../../data/celeba_sr/X/' + i))
            val_set_X = np.array(val_set_X).astype(float) / 255.0
            val_set_Y = self.perceptual_model(val_set_X)
            assert val_set_Y.shape == (10, 4, 4, 512)
            return val_set_X, np.ones(val_set_X.shape[0]), val_set_Y
        except AssertionError as ae:
            print("Val set not loaded correctly")


if __name__ == '__main__':
    srgan = SR_GAN()
    gan_loss: int = 0
    disc_loss: int = 0
    for epoch in range(PARAMS['epochs']):
        print(f"Epoch {epoch}")

        gan_loss, disc_loss = srgan.train_epoch(gan_loss, disc_loss)
        write_logs(srgan.disc_callback, srgan.disc_metrics, logs=disc_loss, epoch_no=epoch)
        write_logs(srgan.gan_callback, srgan.gan_metrics, logs=gan_loss, epoch_no=epoch)

        print(f"Discriminator loss epoch {epoch}: {disc_loss}")
        print(f"Generator loss epoch {epoch}: {gan_loss}")

        gan_loss = 0
        disc_loss = 0

        if epoch % 10 == 0:
            val_loss = srgan.GAN.test_on_batch(x=srgan.val_set[0], y=[srgan.val_set[1], srgan.val_set[2]])
            write_logs(srgan.gan_callback, srgan.val_metrics, logs=val_loss, epoch_no=epoch // 10)
            pred_image = srgan.generator.predict(srgan.val_set[0][np.random.randint(1, 10, size=1)])
            cv2.imwrite(f"../../data/celeba_val/Pred/{epoch}.png", pred_image)

        srgan.GAN.save(f'../../gan_saved_models/GAN/{epoch}.hdf5')
        srgan.generator.save(f'../../gan_saved_models/generator/{epoch}.hdf5')
        srgan.discriminator.save(f'../../gan_saved_models/discriminator/{epoch}.hdf5')

    # TODO: integrate ESRGAN
    # TODO: args parse for loading pre-trained model

# %%
