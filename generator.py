import numpy as np
from tensorflow import keras
import tensorflow as tf
import keras.backend
from keras import Input, layers, Model


def get_model(noise_size):

    inputs = Input(shape=(noise_size,), dtype=tf.float16)
    dense = layers.Dense(8 * 8 * 256, use_bias=False)(inputs)
    act1 = layers.LeakyReLU(0.2)(dense)
    resh1 = layers.Reshape((8, 8, 256))(act1)

    up1 = layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same', activation='LeakyReLU')(resh1)
    bn1 = layers.BatchNormalization()(up1)
    conv1 = layers.Conv2D(filters=256, kernel_size=4, strides=1, padding='same', use_bias=False)(bn1)
    bn2 = layers.BatchNormalization()(conv1)
    act2 = layers.LeakyReLU(0.2)(bn2)

    up2 = layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same', activation='LeakyReLU')(act2)
    bn3 = layers.BatchNormalization()(up2)
    conv2 = layers.Conv2D(filters=128, kernel_size=4, strides=1, padding='same', use_bias=False)(bn3)
    bn4 = layers.BatchNormalization()(conv2)
    act3 = layers.LeakyReLU(0.2)(bn4)

    up3 = layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', activation='LeakyReLU')(act3)
    bn5 = layers.BatchNormalization()(up3)
    conv3 = layers.Conv2D(filters=64, kernel_size=4, strides=1, padding='same', use_bias=False)(bn5)
    bn6 = layers.BatchNormalization()(conv3)
    act4 = layers.LeakyReLU(0.2)(bn6)

    up4 = layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same')(act4)
    bn7 = layers.BatchNormalization()(up4)
    upact4 = layers.LeakyReLU(0.2)(bn7)
    conv4 = layers.Conv2D(filters=3, kernel_size=5, strides=1, padding='same', use_bias=False)(upact4)
    bn5 = layers.BatchNormalization()(conv4)

    outputs = layers.Activation('tanh')(bn5)

    model = Model(inputs, outputs)
    return model
