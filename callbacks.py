from tensorflow import keras
import tensorflow as tf


class GANcallbacks(keras.callbacks.Callback):
    def __init__(self, num_img=3, noise_dim=128):
        self.num_img = num_img
        self.noise_dim = noise_dim

    def on_epoch_end(self, epoch, logs=None):
        noise_vec = tf.random.normal(shape=(self.num_img, self.noise_dim))
        gen_imgs = self.model.generator(noise_vec)
        gen_imgs = (gen_imgs * 127.5) + 127.5

        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(gen_imgs[i])
            img.save("./output/generated_img_%03d_%d.png" % (epoch, i))

