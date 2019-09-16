import os

import cv2
import keras
import keras.layers as L
import numpy as np
from keras import backend as K
from tqdm import tqdm

#from sklearn.model_selection import train_test_split
# load dataset
"""
X = []
Y_im = []

for img in os.listdir('imagenet_64x64'):
    img_path = str('imagenet_64x64input/' + str(img))
    img_path2 = str('imagenet_64x64/' + str(img))
    print(img_path)
    print(img_path2)
    pic1 = cv2.imread(img_path)
    pic2 = cv2.imread(img_path2)
    X.append(pic1)
    Y_im.append(pic2)

X = np.array(X)
Y_im = np.array(Y_im)
X = X.astype(np.float32)
Y_im = Y_im.astype(np.float32)
print(X.shape)
print(Y_im.shape)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_im, test_size=0.1, random_state=0)
"""
X_test = []
y_test = []
for file in tqdm(os.listdir('stl_val_X')):
    img_path = str('stl_val_y/'+str(file))
    img_path2 = str('stl_val_X/'+str(file))
    # print(img_path)
    # print(img_path2)
    pic1 = cv2.imread(img_path)
    pic2 = cv2.imread(img_path2)
    X_test.append(pic2)
    y_test.append(pic1)

X_test = np.array(X_test)
y_test = np.array(y_test)


def batch_generator(X_folder, y_folder, batch_size):
    batch_X = []
    batch_y = []
    while True:
        for img in os.listdir(X_folder):
            img_path = str(X_folder+'/'+str(img))
            img_path2 = str(y_folder+'/'+str(img))
            # print(img_path)
            # print(img_path2)
            pic1 = cv2.imread(img_path)
            pic2 = cv2.imread(img_path2)
            batch_X.append(pic1)
            batch_y.append(pic2)
            if (len(batch_X) == batch_size or img is None):
                yield np.array(batch_X), np.array(batch_y)
                batch_X = []
                batch_y = []


latent_dim = 30
input_img = L.Input(shape=(64, 64, 3))


def sampling(args):
    z_mu, z_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim))
    return z_mu+K.exp(z_sigma)*epsilon


def vae_loss(x, x_decoded_mean):
    xent_loss = keras.losses.mean_squared_error(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_sigma -
                             K.square(z_mu) - K.exp(z_sigma))
    return xent_loss + kl_loss


# encoder
x = L.Conv2D(filters=16, kernel_size=(3, 3),
             activation='sigmoid', padding='same')(input_img)
x = L.Dropout(0.8)(x)
x = L.MaxPool2D(pool_size=(2, 2), padding='same')(x)
x = L.Conv2D(filters=16, kernel_size=(3, 3),
             activation='sigmoid', padding='same')(x)
x = L.Dropout(0.8)(x)
x = L.MaxPool2D(pool_size=(2, 2), padding='same')(x)
x = L.Conv2D(filters=16, kernel_size=(3, 3),
             activation='sigmoid', padding='same')(x)
x = L.Dropout(0.8)(x)
x = L.MaxPool2D(pool_size=(2, 2), padding='same')(x)
x = L.Conv2D(filters=16, kernel_size=(3, 3),
             activation='sigmoid', padding='same')(x)
x = L.Dropout(0.8)(x)
x = L.MaxPool2D(pool_size=(2, 2), padding='same')(x)
x = L.Flatten()(x)
z_mu = L.Dense(latent_dim)(x)
z_sigma = L.Dense(latent_dim)(x)
z = L.Lambda(sampling, output_shape=(latent_dim,))([z_mu, z_sigma])
print(z.shape)
# decoder
y = L.Dense(units=4*4*16, activation='sigmoid')(z)
y = L.Reshape(target_shape=(4, 4, 16))(y)
y = L.Conv2D(filters=16, kernel_size=(3, 3),
             activation='sigmoid', padding='same')(y)
y = L.Dropout(0.8)(y)
y = L.UpSampling2D(size=(2, 2), data_format='channels_last')(y)
y = L.Conv2D(filters=16, kernel_size=(3, 3),
             activation='sigmoid', padding='same')(y)
y = L.Dropout(0.8)(y)
y = L.UpSampling2D(size=(2, 2), data_format='channels_last')(y)
y = L.Conv2D(filters=16, kernel_size=(3, 3),
             activation='sigmoid', padding='same')(y)
y = L.Dropout(0.8)(y)
y = L.UpSampling2D(size=(2, 2), data_format='channels_last')(y)
y = L.Conv2D(filters=16, kernel_size=(3, 3),
             activation='sigmoid', padding='same')(y)
y = L.Dropout(0.8)(y)
y = L.UpSampling2D(size=(2, 2), data_format='channels_last')(y)
y = L.Conv2D(filters=3, kernel_size=(3, 3), padding='same')(y)

vae = keras.models.Model(inputs=input_img, outputs=y)
print(vae.summary())

vae.compile(loss=vae_loss, optimizer=keras.optimizers.adamax(
    0.005), metrics=['accuracy'])

"""
vae.fit(x=X_train, y=Y_train, epochs=30, callbacks=[keras.callbacks.TensorBoard(
), keras.callbacks.ModelCheckpoint('checkpoint.hdf5')], validation_data=(X_test, Y_test), shuffle=True)
"""
vae.fit_generator(batch_generator(str('stl64x64_input'), str('stl64x64'), 95), steps_per_epoch=1000, epochs=30, callbacks=[keras.callbacks.ModelCheckpoint(
    'checkpoint.hdf5'), keras.callbacks.TensorBoard()], validation_data=(X_test, y_test), shuffle=True, use_multiprocessing=True)
vae.save('vae1.h5')
vae.save_weights('vae_weights1.h5')
