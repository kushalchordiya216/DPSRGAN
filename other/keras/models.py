from keras.layers import BatchNormalization, Conv2D, PReLU, Add, LeakyReLU, UpSampling2D, Flatten, Dense, Activation, \
    Input
from keras.models import Model, load_model
from keras.losses import binary_crossentropy, mean_squared_error
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
from keras.backend import mean, square


# ################################################### Generator ######################################################


def ResBlock(x, filters):
    res = x
    x = Conv2D(filters=filters, kernel_size=(
        3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(filters=filters, kernel_size=(
        3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([res, x])
    return x


def CreateGenerator():
    img = Input(shape=(32, 32, 3))
    x = Conv2D(filters=64, kernel_size=(9, 9),
               strides=(1, 1), padding='same')(img)
    x = PReLU(shared_axes=[1, 2])(x)
    res = x
    for i in range(16):
        x = ResBlock(x, 64)
    x = Conv2D(filters=64, kernel_size=(3, 3),
               strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([res, x])
    x = Conv2D(filters=256, kernel_size=(3, 3),
               strides=(1, 1), padding='same')(x)
    x = UpSampling2D()(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(filters=256, kernel_size=(3, 3),
               strides=(1, 1), padding='same')(x)
    x = UpSampling2D()(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(filters=3, kernel_size=(9, 9),
               strides=(1, 1), padding='same', activation='relu')(x)
    gen = Model(inputs=img, outputs=x)
    return gen


Generator = CreateGenerator()


# ################################################ Discriminator ###################################################
def DiscConvBlock(x, filters, kernel_size, strides):
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x


def CreateDiscriminator():
    img = Input(shape=(128, 128, 3))
    x = Conv2D(filters=64, kernel_size=(3, 3),
               strides=(1, 1), padding='same')(img)
    x = DiscConvBlock(x, filters=64, kernel_size=(3, 3), strides=(2, 2))
    x = DiscConvBlock(x, filters=128, kernel_size=(3, 3), strides=(1, 1))
    x = DiscConvBlock(x, filters=128, kernel_size=(3, 3), strides=(2, 2))
    x = DiscConvBlock(x, filters=256, kernel_size=(3, 3), strides=(1, 1))
    x = DiscConvBlock(x, filters=256, kernel_size=(3, 3), strides=(2, 2))
    x = DiscConvBlock(x, filters=512, kernel_size=(3, 3), strides=(1, 1))
    x = DiscConvBlock(x, filters=512, kernel_size=(3, 3), strides=(2, 2))
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    dis = Model(inputs=img, outputs=x)
    dis.compile(loss=binary_crossentropy,
                optimizer=Adam(0.0002, beta_1=0.5, epsilon=1e-8))
    return dis


Discriminator = CreateDiscriminator()


# ################################################ Perception Model ####################################################


def CreatePerceptionNet():
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    vgg.trainable = False
    for l in vgg.layers:
        l.trainable = False
    vggmodel = Model(inputs=vgg.input,
                     outputs=vgg.get_layer('block5_conv4').output)
    vggmodel.trainable = False
    return vggmodel


PerceptionNet = CreatePerceptionNet()


def ContentLoss(y_true, y_pred):
    loss = mean(square(PerceptionNet(y_true) - PerceptionNet(y_pred)))
    return loss


################################################# Combined Model #######################################################


def CreateCombinedGAN(discriminator, generator):
    discriminator.trainable = False
    gan_input = Input(shape=(32, 32, 3))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x, gan_output])
    gan.compile(loss=[ContentLoss, binary_crossentropy], loss_weights=[1, 1e-3],
                optimizer=Adam(0.0002, beta_1=0.5, epsilon=1e-8))
    return gan


GAN = CreateCombinedGAN(Discriminator, Generator)
