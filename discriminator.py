import numpy as np
from tensorflow import keras
import tensorflow as tf
import keras.backend
from keras import Input, layers, Model

class ClipConstraint(keras.constraints.Constraint):
    def __init__(self, clip_val):
        self.clip_val = clip_val

    def __call__(self, weights):
        return keras.backend.clip(weights, -self.clip_val, self.clip_val)

    def get_config(self):
        return {'clip_val': self.clip_val}


def get_model(img_size):
    const = ClipConstraint(0.01)
    inputs = Input(shape=img_size, dtype=tf.float16)

    conv1 = layers.Conv2D(filters=64, kernel_size=5, strides=2, use_bias=True, padding='same',
                          activation=layers.LeakyReLU(0.2))(inputs)
    conv2 = layers.Conv2D(filters=128, kernel_size=5, strides=2, use_bias=True, padding='same',
                          activation=layers.LeakyReLU(0.2))(conv1)
    drop1 = layers.Dropout(0.3)(conv2)
    conv3 = layers.Conv2D(filters=256, kernel_size=5, strides=2, use_bias=True, padding='same',
                          activation=layers.LeakyReLU(0.2))(drop1)
    drop2 = layers.Dropout(0.3)(conv3)
    conv4 = layers.Conv2D(filters=512, kernel_size=5, strides=2, use_bias=True, padding='same',
                          activation=layers.LeakyReLU(0.2))(drop2)
    flat = layers.Flatten()(conv4)
    drop3 = layers.Dropout(0.2)(flat)
    outputs = layers.Dense(1, activation='linear')(drop3)
    model = Model(inputs, outputs)
    return model