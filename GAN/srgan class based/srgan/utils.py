"""
Includes utilities for building srgan, like the generator and discriminator blocks
"""
import os
import random

import cv2
import numpy as np

from keras.applications import VGG19
from keras.callbacks import TensorBoard
from tensorflow import Summary
from keras.layers import Conv2D, Conv2DTranspose, PReLU, LeakyReLU, BatchNormalization
from keras.models import Model
from keras.utils import plot_model

from .resnet_arch import RRDB

PARAMS = {'n_latent': 100, 'batch_size': 32, 'epochs': 50, 'steps_per_epoch': 200,
          'path_input': '/home/kushal/WorkSpace/Python/data/celeba_sr/celeba_resized_inp/',
          'path_output': '/home/kushal/WorkSpace/Python/data/celeba_sr/celeba_resized/',
          'disc_looping_factor': 2}


def generator_block(input_image, filters=8):
    """
    Takes an image and puts it though a Conv->RRDB->ConvTranspose architecture

    :param input_image: matrix with dimensions (h,w,c)
    :return: matrix with the same dimensions as input matrix
    """
    x = Conv2D(filters=filters, kernel_size=(
        5, 5), padding='valid')(input_image)
    x = PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(x)
    x = RRDB(x, filters)
    x = Conv2DTranspose(filters=filters, kernel_size=(
        5, 5), padding='valid')(x)
    x = PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(x)
    return x


def discriminator_block(inputs, filters=8):
    x = Conv2D(kernel_size=(5, 5), filters=filters, padding='valid',
               input_shape=(64, 64, 3))(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    return x


def image_loader():
    """
    """
    image_x = []
    image_y = []
    image_list = random.sample(os.listdir(PARAMS['path_output']), PARAMS['batch_size'])
    for i in image_list:
        image_x.append(cv2.imread(PARAMS['path_input'] + str(i)))
        image_y.append(cv2.imread(PARAMS['path_output'] + str(i)))
    return np.array(image_x).astype(float) / 255.0, np.array(image_y).astype(float) / 255.0


def build_perceptual() -> Model:
    vgg_model: Model = VGG19(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
    for layer in vgg_model.layers:
        layer.trainable = False
    outputs = [vgg_model.get_layer('block5_conv4').output, vgg_model.get_layer('block4_conv4').output,
               vgg_model.get_layer('block3_conv4').output, vgg_model.get_layer('block2_conv2').output]
    perceptual_model = Model(inputs=vgg_model.input, outputs=outputs)
    perceptual_model.name = "PerceptualModel"
    plot_model(model=perceptual_model,to_file="srgan/created_models/vgg19_perceptual_model.png")
    print(perceptual_model.summary())
    return perceptual_model


def write_logs(callback: TensorBoard, names, logs, epoch_no):
    for name, value in zip(names, logs):
        summary = Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, epoch_no)
        callback.writer.flush()
