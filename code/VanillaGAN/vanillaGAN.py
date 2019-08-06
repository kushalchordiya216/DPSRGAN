import tensorflow as tf
from resnet_arch import RRDB
from data_loader import DataPipeline
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
L = tf.keras.layers
M = tf.keras.models
PARAMS = {'n_latent': 100, 'batch_size': 128, 'epochs': 50, 'shuffle_size': 5000, 'prefetch_size': 256,
          'path': '/home/kushal/WorkSpace/Python/Super-Resolution/data/valid_64x64/',
          'shape': [64, 64, 3], 'looping_factor': 2}


class VanillaGAN(tf.keras.Model):
    def compute_output_signature(self, input_signature):
        # only added cause its inheriting from tf.keras.Model
        # so it has to implement all abstract methods
        pass

    def __init__(self, *args, **kwargs):
        super(VanillaGAN, self).__init__()
        self.n_latent = PARAMS['n_latent']
        self.epochs = PARAMS['epochs']
        self.batch_size = PARAMS['batch_size']
        self.generator: tf.keras.models.Model = self.build_generator()
        self.discriminator: tf.keras.models.Model = self.build_discriminator()

        # since two networks are being trained simultaneously we have to use two different optimizers,
        # so as to not interfere with things like lr-decay or weight-decay

        self.gen_optimizer = tf.keras.optimizers.Adam(0.0001)
        self.disc_optimizer = tf.keras.optimizers.Adam(0.0001)
        self.checkpoint_dir = "../../GAN_savedModels"
        self.checkpoint_prefix = "vgan.ckpt"
        self.Checkpoint = tf.train.Checkpoint(generator_optimizer=self.gen_optimizer,
                                              discriminator_optimizer=self.disc_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        self.generator.summary()
        self.discriminator.summary()

    def build_generator(self) -> tf.keras.models.Model:
        random_noise = L.Input(shape=(self.n_latent,))
        x = L.Dense(54 * 54 * 3, activation='relu')(random_noise)
        x = L.Reshape((54, 54, 3))(x)
        x = L.Conv2DTranspose(filters=8, kernel_size=(
            5, 5), padding='valid')(x)
        x = RRDB(x, 'relu', 8)
        x = L.Conv2DTranspose(filters=8, kernel_size=(
            5, 5), padding='valid')(x)
        x = RRDB(x, 'relu', 8)
        x = L.Conv2DTranspose(filters=16, kernel_size=(
            5, 5), padding='valid')(x)
        x = RRDB(x, 'relu', 16)
        x = L.Conv2D(filters=3, kernel_size=(3, 3), padding='valid')(x)
        g_model = tf.keras.models.Model(random_noise, x)
        return g_model

    def build_discriminator(self) -> tf.keras.models.Model:
        d_model = M.Sequential()
        d_model.add(
            L.Conv2D(kernel_size=(5, 5), activation='relu', filters=4, padding='valid',
                     input_shape=(64, 64, 3)))
        d_model.add(L.BatchNormalization())
        d_model.add(L.Conv2D(kernel_size=(5, 5), activation='relu', filters=8, padding='valid'))
        d_model.add(L.MaxPool2D((2, 2), padding='valid'))
        d_model.add(L.BatchNormalization())

        d_model.add(L.Conv2D(kernel_size=(5, 5), activation='relu', filters=16, padding='valid'))
        d_model.add(L.BatchNormalization())
        d_model.add(L.Conv2D(kernel_size=(5, 5), activation='relu', filters=32, padding='valid'))
        d_model.add(L.MaxPool2D((2, 2), padding='valid'))
        d_model.add(L.BatchNormalization())
        d_model.add(L.Flatten())
        d_model.add(L.Dense(1))
        return d_model

    @staticmethod
    def loss(labels, pred):
        return tf.keras.losses.binary_crossentropy(labels, pred, from_logits=True)

    def discriminator_loss(self, real_pred: tf.Tensor, generated_pred: tf.Tensor):
        real_loss = self.loss(tf.ones_like(real_pred), real_pred)
        gen_loss = self.loss(tf.zeros_like(generated_pred), generated_pred)
        return 0.5 * (real_loss + gen_loss)

    def generator_loss(self, generated_pred: tf.Tensor):
        return self.loss(tf.ones_like(generated_pred.shape), generated_pred)

    @tf.function
    def train(self, images):
        for _ in range(PARAMS['looping_factor']):  # train discriminator for n loops
            noise = tf.random.normal(shape=[self.batch_size, 100], mean=0, stddev=1.0, dtype=tf.float32)
            with tf.GradientTape() as disc_tape:
                gen_images = self.generator(noise)
                real_pred = self.discriminator(images)
                generated_pred = self.discriminator(tf.zeros_like(gen_images))
                # generator_loss = self.generator_loss(generated_pred)
                discriminator_loss = self.discriminator_loss(real_pred, generated_pred)
                # avg_gen_loss.update_state(generator_loss)
                avg_disc_loss.update_state(discriminator_loss)
            disc_gradients = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients((zip(disc_gradients, self.discriminator.trainable_variables)))

        # train generator once
        noise = tf.random.normal(shape=[self.batch_size, 100], mean=0, stddev=1.0, dtype=tf.float32)
        with tf.GradientTape() as gen_tape:
            gen_images = self.generator(noise)
            generated_pred = self.discriminator(gen_images)
            generator_loss = self.generator_loss(generated_pred)
            avg_gen_loss.update_state(generator_loss)
        gen_gradients = gen_tape.gradient(generator_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

    def predict(self, x, epoch):
        predictions = self.generator(x)
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        plt.savefig('./predicted_Images/image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()

    def save_model(self, epoch):
        manager = tf.train.CheckpointManager(self.Checkpoint, self.checkpoint_dir, max_to_keep=5)
        print(f'saving checkpoints on epoch {epoch} ')
        self.Checkpoint.save(file_prefix=self.checkpoint_prefix)


if __name__ == '__main__':
    Vanilla_GAN = VanillaGAN()
    d_p = DataPipeline(path=PARAMS['path'], batch_size=PARAMS['batch_size'],
                       shuffle_size=PARAMS['shuffle_size'], prefetch_buffer=PARAMS['prefetch_size'])
    input("Continue?[y/n]")
    cur_disc_loss = 100000
    cur_gen_loss = 100000
    train_dataset = d_p.pipe_lining(d_p.read_dataset())
    avg_disc_loss = tf.keras.metrics.Mean()
    avg_gen_loss = tf.keras.metrics.Mean()

    for epoch in range(Vanilla_GAN.epochs):
        for batch in tqdm(train_dataset):
            Vanilla_GAN.train(batch)
            print(f"for epoch {epoch} gen_loss is {avg_gen_loss} disc_loss is {avg_disc_loss}")

        if avg_gen_loss.result() < cur_gen_loss or avg_disc_loss.result() < cur_disc_loss:
            Vanilla_GAN.save_model(epoch)
            cur_disc_loss = avg_disc_loss.result()
            cur_gen_loss = avg_gen_loss.result()
        avg_disc_loss.reset_states()
        avg_gen_loss.reset_states()
        seed = tf.random.normal([10, Vanilla_GAN.n_latent])
        if epoch % 10 == 0:
            Vanilla_GAN.predict(x=seed, epoch=epoch)
