# %%
import os
import cv2
import tensorflow as tf
import keras 
import keras.layers as L
import numpy as np
from sklearn.model_selection import train_test_split

# %%
def make_model():
    model = keras.models.Sequential()
    model.add(L.Conv2D(filters=32, kernel_size=(3, 3),activation='relu',
                       input_shape=(480, 853, 3), padding='same'))
    model.add(L.BatchNormalization())
    model.add(L.MaxPool2D(pool_size=(2,2),padding='same'))
    model.add(L.Conv2D(filters=16, kernel_size=(3, 3), padding='same',activation='relu'))
    model.add(L.BatchNormalization())    
    model.add(L.MaxPool2D(pool_size=(2,2),padding='same'))
    model.add(L.Conv2D(filters=16, kernel_size=(3, 3), padding='same',activation='relu'))
    
    model.add(L.Conv2D(filters=8, kernel_size=(3, 3), padding='same',activation='relu'))
    model.add(L.Conv2D(filters=8, kernel_size=(3, 3), padding='same',activation='relu'))

    model.add(L.Conv2D(filters=8, kernel_size=(3, 3), padding='same',activation='relu'))
    model.add(L.Conv2D(filters=8, kernel_size=(3, 3), padding='same',activation='relu'))

    model.add(L.Conv2D(filters=16, kernel_size=(3, 3), padding='same',activation='relu'))
    model.add(L.BatchNormalization())    
    model.add(L.Conv2D(filters=16, kernel_size=(3, 3), padding='same',activation='relu'))
    model.add(L.BatchNormalization())
    model.add(L.UpSampling2D(size=(2,2),data_format="channels_last"))
    model.add(L.Conv2D(filters=32, kernel_size=(3, 3), padding='same',activation='relu'))
    model.add(L.BatchNormalization())
    model.add(L.UpSampling2D(size=(2,2),data_format="channels_last"))
    model.add(L.Conv2D(filters=3, kernel_size=(3, 3), padding='same',activation='relu'))
    model.add(L.BatchNormalization())
    model.add(L.UpSampling2D(size=(2,2),data_format="channels_last"))

    return model

# %%
'''Load datasets'''
y_train = []
X_train = []
X_test = []
y_test = []

# path = ('./Linnaeus 5 32x32/train/')
# path2 = ('./Linnaeus 5 64x64/train/')
# path_test = ('./Linnaeus 5 32x32/test/')
# path2_test = ('./Linnaeus 5 64x64/test/')


# for [(r1,d1,f1),(r2,d2,f2),(r3,d3,f3),(r4,d4,f4)] in zip(os.walk(path),os.walk(path2),os.walk(path_test),os.walk(path2_test)):
#     for files in f1:
#         X_train.append(cv2.imread(r1 + '/' + files))
#     for files in f2:
#         y_train.append(cv2.imread(r2 + '/' + files))
#     for files in f3:
#         X_test.append(cv2.imread(r3 + '/' + files))
#     for files in f4:
#         y_test.append(cv2.imread(r4 + '/' + files))

path = ('./data/480/')
path2 = ('./data/720/')
for file1 in os.listdir(path):
    X_train.append(cv2.imread(path+file1))
    y_train.append(cv2.imread(path2+file1))

X_test = X_train[int(len(X_train)*0.9):]
y_test = y_train[int(len(y_train)*0.9):]
X_train =  X_train[:int(len(X_train)*0.9)]
y_train =  y_train[:int(len(y_train)*0.9)]

print(len(X_train),len(y_train),len(X_test),len(y_test))
# %%
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print(X_train.shape, y_train.shape, X_test.shape,y_test.shape)

# %%
model = make_model()
model.summary()
# %%
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.adam(lr=0.005),
              metrics=['accuracy'])
# %%
from keras.callbacks import TensorBoard
model.fit(X_train, y_train, batch_size=100, epochs=100, callbacks=[
          keras.callbacks.ModelCheckpoint('checkpoint2.hdf5'),TensorBoard(log_dir='/SR/new1/autoencoder')],
          validation_split=0.2, shuffle=True)
# %%
model.evaluate()
# %%
model.save_weights("weights2.h5")
model.save("model2.h5")
