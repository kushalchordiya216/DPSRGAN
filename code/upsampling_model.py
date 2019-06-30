# %%
print('Importing libraries ....')

import keras
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, UpSampling2D
from keras.models import Sequential
import numpy as np
import cv2
import os
from tqdm import tqdm
# %%


def make_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                     input_shape=(32, 32, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    
    model.add(Conv2D(filters=16, kernel_size=(3, 3),
                     padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    
    model.add(Conv2D(filters=16, kernel_size=(3, 3),
                     padding='same', activation='relu'))

    model.add(Conv2D(filters=8, kernel_size=(
        3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=8, kernel_size=(
        3, 3), padding='same', activation='relu'))

    model.add(Conv2D(filters=8, kernel_size=(
        3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=8, kernel_size=(
        3, 3), padding='same', activation='relu'))

    
    model.add(Conv2D(filters=16, kernel_size=(
        3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=(
        3, 3), padding='same', activation='relu'))
    
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2), data_format="channels_last"))
    
    model.add(Conv2D(filters=32, kernel_size=(
        3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2), data_format="channels_last"))
    
    model.add(Conv2D(filters=3, kernel_size=(
        3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2), data_format="channels_last"))
    return model


# %%
y_train = []
X_train = []
X_test = []
y_test = []

path = ('./data/valid_32x32/')
path2 = ('./data/valid_64x64/')
path_test = ('./data/Linnaeus 5 32x32/train/')
path2_test = ('./data/Linnaeus 5 64x64/train/')


print('Loading training dataset ....')
for f1 in tqdm(os.listdir(path)):
    X_train.append(cv2.imread(path + f1))
    y_train.append(cv2.imread(path2 + f1))

print('''Loading test datasets ....''')
for [(r1, d1, f1), (r2, d2, f2)] in tqdm(zip(os.walk(path_test), os.walk(path2_test))):
    for files in f1:
        X_test.append(cv2.imread(r1 + '/' + files))
    for files in f2:
        y_test.append(cv2.imread(r2 + '/' + files))

# %%
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# %%
model = make_model()
model.summary()
# %%
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.adam(lr=0.005),
              metrics=['accuracy'])
# %%
model.fit(X_train, y_train, batch_size=100, epochs=100, callbacks=[
          keras.callbacks.ModelCheckpoint('upsampling saved models/checkpoint3.hdf5'), TensorBoard(log_dir='/SR/deep/autoencoder')],
          validation_split=0.2, shuffle=True)

# model.fit_generator()
# %%
model.evaluate(x=X_test, y=y_test, batch_size=100, verbose=1)
# %%
model.save_weights("upsampling saved models/weights3.h5")
model.save("upsampling saved models/model3.h5")
