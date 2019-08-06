# %%
import tensorflow as tf

K = tf.keras.backend
Layer = tf.keras.layers.Layer

L = tf.keras.layers
A = tf.keras.activations
M = tf.keras.models


# %%
def resBlock(input_tensor, activation, n_filters):
    conv1 = L.Conv2DTranspose(filters=n_filters, kernel_size=(
        3, 3), padding='same', activation=activation)(input_tensor)
    conv2 = L.Conv2DTranspose(filters=n_filters, kernel_size=(
        3, 3), padding='same', activation=activation)(conv1)
    input_3 = L.add([conv2, conv1])
    conv3 = L.Conv2DTranspose(filters=n_filters, kernel_size=(
        3, 3), padding='same')(input_3)
    input_4 = L.add([conv3, input_3, conv1])
    final_layer = L.Conv2DTranspose(filters=n_filters, padding='same',
                                    kernel_size=(3, 3), activation=activation)(input_4)
    return final_layer


# %%

def RRDB(input_tensor, activation, n_filters):
    """
    Residual within residual block.
    The function is a Convolutional ResNet consisting on three blocks which are all residual blocks in turn
    to refer to the original residual block go to resBlock

    :param input_tensor: the tensor which is to be given as input to the RRDB
    must have shape = (batch_size,n_H,n_W,n_C)

    :param activation: The activation functions to ber applied through the ResBlock.
    relu activation is recommended

    :param n_filters: no of filters per conv block.
    Filters are uniform across all conv blocks.
    More filters allow network to learn more features, at the cost of more computation.

    :returns final_layer: the output layer of the RRDB, Has the same shape as input_tensor
    """
    output1 = resBlock(input_tensor, activation, n_filters)
    input2 = L.add([input_tensor, output1])
    output2 = resBlock(input2, activation, n_filters)
    input3 = L.add([input_tensor, input2, output2])
    output3 = resBlock(input3, activation, n_filters)
    return output3


