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

    up1 = layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same')(resh1)
    conv1 = layers.Conv2D(filters=256, kernel_size=4, strides=1, padding='same', use_bias=False)(up1)
    bn2 =layers.BatchNormalization()(conv1)
    act2 = layers.LeakyReLU(0.2)(bn2)

    up2 = layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same')(act2)
    conv2 = layers.Conv2D(filters=128, kernel_size=4, strides=1, padding='same', use_bias=False)(up2)
    bn3 = layers.BatchNormalization()(conv2)
    act3 = layers.LeakyReLU(0.2)(bn3)

    up3 = layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same')(act3)
    conv3 = layers.Conv2D(filters=64, kernel_size=4, strides=1, padding='same', use_bias=False)(up3)
    bn4 = layers.BatchNormalization()(conv3)
    act4 = layers.LeakyReLU(0.2)(bn4)

    up4 = layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same')(act4)
    upact4 = layers.LeakyReLU(0.2)(up4)
    conv4 = layers.Conv2D(filters=3, kernel_size=5, strides=1, padding='same', use_bias=False)(upact4)
    bn5 = layers.BatchNormalization()(conv4)

    outputs = layers.Activation('tanh')(bn5)

    model = Model(inputs, outputs)
    return model
