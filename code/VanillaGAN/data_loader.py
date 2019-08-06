# %%
import tensorflow as tf
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataPipeline:
    def __init__(self, path, batch_size=128, shuffle_size=5000, prefetch_buffer=AUTOTUNE):
        self.path = path
        self.batch_size = batch_size
        self.shuffle_size = shuffle_size
        self.prefetch_buffer = prefetch_buffer

    def preprocess_image(self):

        pass

    def read_image(self, img_path):
        image_raw = tf.io.read_file(img_path)
        image = tf.image.decode_png(image_raw)
        return image

    def process_paths(self):
        true_path = []
        for name in os.listdir(self.path):
            true_path.append(self.path + name)
        return true_path

    def read_dataset(self):
        true_path = self.process_paths()
        img_paths = tf.data.Dataset.from_tensor_slices(true_path)
        image_ds = img_paths.map(self.read_image, num_parallel_calls=AUTOTUNE)
        return image_ds

    def pipe_lining(self, image_ds):
        ds = image_ds.shuffle(buffer_size=5000)
        ds = ds.repeat()
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=self.prefetch_buffer)
        return ds
