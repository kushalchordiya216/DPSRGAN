# VAE model written with tf2.0
# %%
import tensorflow as tf
import cv2
import os
import time
from tqdm import tqdm
import numpy as np
# %%

FLAGS = {'batch_size': 128, 'epochs': 30, 'shuffle_buffer_size': 100}


class VAE(tf.keras.Model):
    def __init__(self, n_latent):
        super(VAE, self).__init__()
        self.n_latent = n_latent
        self.encoder_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(filters=8, kernel_size=(
                3, 3), padding='valid', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(
                3, 3), padding='valid', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(
                3, 3), padding='valid', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(
                3, 3), padding='valid', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(
                3, 3), padding='valid', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.n_latent*2),
        ])
        # print(self.encoder_net.summary())
        self.decoder_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.n_latent,)),
            tf.keras.layers.Dense(54*54*8, activation='relu'),
            tf.keras.layers.Reshape(target_shape=(54, 54, 8)),
            tf.keras.layers.Conv2DTranspose(
                filters=16, kernel_size=(3, 3), activation='softmax', padding='valid'),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv2DTranspose(
                filters=16, kernel_size=(3, 3), activation='softmax', padding='valid'),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv2DTranspose(
                filters=16, kernel_size=(3, 3), activation='softmax', padding='valid'),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv2DTranspose(
                filters=16, kernel_size=(3, 3), activation='softmax', padding='valid'),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv2DTranspose(
                filters=3, kernel_size=(3, 3), activation='softmax', padding='valid')
        ])
        # print(self.decoder_net.summary())

    def rand_sampling(self, test_data):
        if test_data is None:
            test_data = tf.random.normal(shape=(50, self.n_latent), stddev=1.0)
        return decode(test_data)

    def encode(self, input_data):
        mean, log_variance = tf.split(
            self.encoder_net(input_data), num_or_size_splits=2, axis=1)
        return mean, log_variance

    def reparametrize(self, mean, log_variance):
        epsilon = tf.random.normal(shape=mean.shape, stddev=1.0)
        return epsilon*tf.exp(log_variance) + mean

    def decode(self, z):
        generated = self.decoder_net(z)
        return generated


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def loss_function(model, input_data, ground_truth):
    mean, log_variance = model.encode(input_data)
    z = model.reparametrize(mean, log_variance)
    generated = model.decode(z)
    loss = tf.keras.losses.MSE(ground_truth, generated) + \
        tf.keras.losses.kld(ground_truth, generated)
    avg_acc.update_state(ground_truth, generated)
    avg_loss.update_state(loss)
    return loss


def gradient_calc(model: VAE, input_data, ground_truth):
    with tf.GradientTape() as tape:
        loss = loss_function(
            model=model, input_data=input_data, ground_truth=ground_truth)
    return tape.gradient(loss, model.trainable_weights), loss


def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))
# %%


def data_loader_generator():
    tf.data.Dataset.list_files('./data/valid_32x32/*.png')
    path = "./data/valid_32x32/"
    path2 = "./data/valid_64x64/"
    input_data = []
    output_data = []
    while True:
        for img in os.listdir(path):
            input_data.append(cv2.imread(path+img))
            output_data.append(cv2.imread(path2+img))
            if len(input_data) == FLAGS['batch_size']:
                input_data = np.asarray(input_data)
                output_data = np.asarray(output_data)
                output_data = output_data.astype(np.float32)
                input_data = input_data.astype(np.float32)
                yield input_data, output_data
                input_data = []
                output_data = []

# %%


# %%
if __name__ == "__main__":

    model = VAE(30)
    optimizer = tf.optimizers.Adam(1e-4)
    input_data = data_loader_generator()
    for epoch in range(FLAGS['epochs']):
        start_time = time.time()
        avg_loss = tf.keras.metrics.Mean()
        avg_acc = tf.keras.metrics.BinaryAccuracy()
        for inputs, outputs in tqdm(input_data):
            gradients, loss = gradient_calc(model, input_data=inputs,
                                            ground_truth=outputs)
            apply_gradients(optimizer=optimizer, gradients=gradients,
                            variables=model.trainable_variables)
        end_time = time.time()
        print(
            f"For Epoch {epoch} average loss is {avg_loss.result().numpy()}, average accuracy is {avg_acc.result().numpy()}")
        # tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
        # tf.summary.scalar('acc', avg_acc.result(), step=optimizer.iterations)
        avg_loss.reset_states()
        avg_acc.reset_states()
        print("time for this epoch : ", end_time - start_time)


# %%print(dir(VAE))
