import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

tf.reset_default_graph()
X_in = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, 64 * 64 * 3])
l_test = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='l_test')
pred_test = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='pred_test')
l_train = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='l_train')
pred_train = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='pred_train')
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
n_latent = 512
num_epochs = 10

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
    if len(X) == 100:
        break

X = np.array(X)
Y_im = np.array(Y_im)
X = X.astype(np.float32)
Y_im = Y_im.astype(np.float32)
print(X.shape)
print(Y_im.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_im, test_size=0.1, random_state=0)


def encoder(X_in, keep_prob):
    with tf.variable_scope("encoder", reuse=None):
        x = tf.layers.conv2d(X_in, filters=16, kernel_size=3, strides=1, padding='same', activation=tf.nn.softmax)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=1, padding='same', activation=tf.nn.softmax)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=1, padding='same', activation=tf.nn.softmax)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=1, padding='same', activation=tf.nn.softmax)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent)
        sd = 0.5 * tf.layers.dense(x, units=n_latent)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z = mn + tf.multiply(epsilon, tf.exp(sd))

        return z, mn, sd


def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=64 * 64 * 16)
        x = tf.reshape(x, [-1, 64, 64, 16])
        x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=3, strides=1, padding='same',
                                       activation=tf.nn.softmax)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=3, strides=1, padding='same',
                                       activation=tf.nn.softmax)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=3, strides=1, padding='same',
                                       activation=tf.nn.softmax)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=3, strides=1, padding='same',
                                       activation=tf.nn.softmax)
        x = tf.nn.dropout(x, keep_prob)
        img = tf.layers.conv2d_transpose(x, filters=3, kernel_size=3, strides=1, padding='same')
        return img


sampled, mn, sd = encoder(X_in, keep_prob)
img = decoder(sampled, keep_prob)

flat = tf.reshape(img, [-1, 64 * 64 * 3])
img_loss = tf.reduce_sum(tf.squared_difference(flat, Y_flat), 1)
kld = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + kld)
optimizer = tf.train.AdamOptimizer(0.005).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

for i in range(num_epochs):
    print("Epoch %d" % int(i + 1))
    _, loss_val = sess.run([optimizer, loss], feed_dict={X_in: X_train, Y: Y_train, keep_prob: 0.8})
    print("Val loss = %s" % loss_val)
    """
    if not i % 2:
        init1 = tf.global_variables_initializer()
        sess.run(init1)
        init2 = tf.local_variables_initializer()
        sess.run(init2)
        Y_pred_test = sess.run(img, feed_dict={X_in: X_test, keep_prob: 1})
        test_acc, test_op = tf.metrics.accuracy(l_test, pred_test)
        print(sess)
        print("Test acc")
        print(sess.run([test_acc, test_op], feed_dict={l_test: Y_test, pred_test: Y_pred_test}))
        Y_pred_train = sess.run(img, feed_dict={X_in: X_train, keep_prob: 1})
        train_acc, train_op = tf.metrics.accuracy(l_train, pred_train)
        print("Train acc")
        print(sess.run([train_acc, train_op], feed_dict={l_train: Y_train, l_test: Y_pred_train}))
        """
    
savePath = saver.save(sess, 'vaemodel1.ckpt')

"""
image = cv2.imread('picc.png')
image = np.array(image)
image = image.astype(np.float32)
image = np.reshape(image, (1, 64, 64, 3))
print(image.shape)
pred = sess.run(img, feed_dict={X_in: image, keep_prob: 1})
pred = np.reshape(pred, (64, 64, 3))
cv2.imwrite('predvae.png', pred)
"""
