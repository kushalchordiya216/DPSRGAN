import os
from keras import layers as L
import cv2
import numpy as np
from keras.models import Model
import keras
from tqdm import tqdm

epochs = 400
batch_size = 200
dvsgr = 2
steps_per_epoch = 200


# load data
print("Loading images")
X_train = []
y_train = []
iterator = tqdm(os.listdir('../../celeba_resized'))
for file in iterator:
    img_path1 = str('../../celeba_resized/'+str(file))
    img_path2 = str('../../celeba_resized_inp/'+str(file))
    pic1 = cv2.imread(img_path1)
    pic2 = cv2.imread(img_path2)
    X_train.append(pic2)
    y_train.append(pic1)
    if len(X_train) == 1000:
        iterator.close()
        break

X_test = []
for file in os.listdir('../celeba_val/X'):
    img_path = str('../celeba_val/X/'+str(file))
    pic = cv2.imread(img_path)
    X_test.append(pic)

X_train = np.array(X_train)
X_train = X_train.astype(float)/255.0
y_train = np.array(y_train)
y_train = y_train.astype(float)/255.0
X_test = np.array(X_test)
X_test = X_test.astype(float)/255.0


def ResBlock(x, filters):
    res = x
    x = L.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = L.ReLU()(x)
    x = L.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = L.ReLU()(x)
    x = L.Add()([x, res])
    return x


def create_generator():
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
    x = L.Dropout(0.2)(x)

    x = L.Conv2D(filters=16, kernel_size=(3, 3),
                 padding='same', activation='relu')(x)
    x = L.BatchNormalization()(x)
    x = ResBlock(x, 16)
    x = L.BatchNormalization()(x)
    x = ResBlock(x, 16)
    x = L.BatchNormalization()(x)
    x = ResBlock(x, 16)
    x = L.BatchNormalization()(x)
    x = L.Dropout(0.2)(x)

    x = L.Conv2D(filters=32, kernel_size=(3, 3),
                 padding='same', activation='relu')(x)
    x = L.BatchNormalization()(x)
    x = ResBlock(x, 32)
    x = L.BatchNormalization()(x)
    x = ResBlock(x, 32)
    x = L.BatchNormalization()(x)
    x = ResBlock(x, 32)
    x = L.BatchNormalization()(x)
    x = L.Dropout(0.2)(x)

    x = L.Conv2D(filters=3, kernel_size=(3, 3),
                 padding='same', activation='sigmoid')(x)
    gen = Model(inputs=img, outputs=x)
    gen.compile(loss=keras.losses.binary_crossentropy,
                optimizer=keras.optimizers.adam(0.001))
    return gen


def create_discriminator():
    inp = L.Input(shape=(64, 64, 3))
    x = L.Conv2D(filters=8, kernel_size=(3, 3), padding='valid')(inp)
    x = L.ReLU()(x)
    x = L.Dropout(0.2)(x)
    x = L.Conv2D(filters=16, kernel_size=(3, 3), padding='valid')(x)
    x = L.ReLU()(x)
    x = L.Dropout(0.2)(x)
    x = L.Conv2D(filters=32, kernel_size=(3, 3), padding='valid')(x)
    x = L.ReLU()(x)
    x = L.Dropout(0.2)(x)
    x = L.Flatten()(x)
    x = L.Dense(1, activation='sigmoid')(x)
    dis = Model(inputs=inp, outputs=x)
    dis.compile(loss=keras.losses.binary_crossentropy,
                optimizer=keras.optimizers.adam(0.001))
    return dis


def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = L.Input(shape=(64, 64, 3))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss=keras.losses.binary_crossentropy,
                optimizer=keras.optimizers.adam(0.001))
    return gan


generator = create_generator()
discriminator = create_discriminator()
gan = create_gan(discriminator, generator)

print(gan.summary())

for e in range(1, epochs+1):
    print("Epoch %d" % e)
    for _ in tqdm(range(steps_per_epoch)):
        sampled_inputs = X_train[np.random.randint(
            low=0, high=X_train.shape[0], size=batch_size)]
        generated_images = generator.predict(sampled_inputs)
        image_batch = y_train[np.random.randint(
            low=0, high=y_train.shape[0], size=batch_size)]
        X = np.concatenate([image_batch, generated_images])
        y_dis = np.zeros(2*batch_size)
        y_dis[:batch_size] = 1
        discriminator.trainable = True
        for _ in range(dvsgr):
            discriminator_loss = discriminator.train_on_batch(X, y_dis)
        sampled_inputs = X_train[np.random.randint(
            low=0, high=X_train.shape[0], size=batch_size)]
        y_gen = np.ones(batch_size)
        discriminator.Trainable = False
        gan_loss = gan.train_on_batch(sampled_inputs, y_gen)
    pr = generator.predict(
        X_test[np.random.randint(low=0, high=X_test.shape[0], size=2)])
    pr = (pr*255.0).astype(int)
    for i in range(2):
        cv2.imwrite('../celeba_val/pred/pr' +
                    str(e)+'('+str(i)+')'+'.jpg', pr)
    print("Discriminator loss="+str(discriminator_loss))
    print("GAN loss="+str(gan_loss))
    if e % 5 == 0:
        generator.save('../checkpoint_models/generatorep'+str(e)+'.hdf5')
        discriminator.save(
            '../checkpoint_models/discriminatorep'+str(e)+'.hdf5')
generator.save('generator.hdf5')
discriminator.save('discriminator.hdf5')
