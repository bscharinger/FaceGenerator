import generator
import discriminator
from tensorflow import keras
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.client import device_lib
import GAN
import callbacks

dataset = keras.preprocessing.image_dataset_from_directory("./data/img_align_celeba/img_align_celeba/", label_mode=None,
                                                           image_size=(128, 128), batch_size=128)
dataset = dataset.map(lambda x: -1 + ((x*2)/255.0))

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_lib.list_local_devices()

policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)

discriminator_mod = discriminator.get_model((128, 128, 3))
generator_mod = generator.get_model(256)
discriminator_mod.summary()
generator_mod.summary()

d_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9, clipvalue=1)
g_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)


def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


epochs = 30


gan = GAN.GAN(discriminator_mod, generator_mod, 256, disc_extra_steps=1)
gan.compile(d_optimizer=d_optimizer, g_optimizer=g_optimizer,
            g_loss_func=generator_loss, d_loss_func=discriminator_loss)

history = gan.fit(dataset, epochs=epochs,
                  callbacks=[callbacks.GANcallbacks(10, 256),
                             keras.callbacks.TensorBoard(log_dir="./logs", update_freq='batch')])

gan.save('GAN_FaceGen_model_30epochs')
