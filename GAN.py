from tensorflow import keras
import tensorflow as tf


class GAN(keras.Model):
    def __init__(self, discriminator, generator, noise_dim, disc_extra_steps, gp_weight=10.0):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.noise_dim = noise_dim
        self.disc_extra_steps = disc_extra_steps
        self.gp_weight = gp_weight
        self.built = True


    def compile(self, d_optimizer, g_optimizer, d_loss_func, g_loss_func):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_func = d_loss_func
        self.g_loss_func = g_loss_func
        self.d_loss_metric = keras.metrics.Mean(name="Discr. Loss")
        self.g_loss_metric = keras.metrics.Mean(name="Gen. Loss")

    def gradient_penalty(self, batch_size, real_imgs, fake_imgs):
        alpha = tf.cast(tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0), dtype=tf.float16)
        diff = tf.cast(fake_imgs, dtype=tf.float16) - tf.cast(real_imgs, dtype=tf.float16)
        interpolated = real_imgs + alpha * tf.cast(diff, dtype=tf.float16)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @tf.function
    def train_step(self, real_images):
        # Sample noise data
        batch_size = tf.shape(real_images)[0]

        # discriminator training
        for i in range(self.disc_extra_steps):
            noise_vector = tf.random.normal(shape=(batch_size, self.noise_dim), dtype=tf.float16)
            with tf.GradientTape() as tape:
                fake_images = self.generator(noise_vector, training=True)
                fake_logits = self.discriminator(fake_images, training=True)
                real_logits = self.discriminator(real_images, training=True)

                d_cost = self.d_loss_func(real_logits, fake_logits)
                gp = self.gradient_penalty(batch_size, tf.cast(real_images, dtype=tf.float16), fake_images)
                d_loss = d_cost + gp * self.gp_weight

            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        # generator training
        noise_vector = tf.random.normal(shape=(batch_size, self.noise_dim), dtype=tf.float16)
        with tf.GradientTape() as tape:
            generated_imgs = self.generator(noise_vector, training=True)
            gen_imgs_logits = self.discriminator(generated_imgs, training=True)
            g_loss = self.g_loss_func(gen_imgs_logits)
        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

