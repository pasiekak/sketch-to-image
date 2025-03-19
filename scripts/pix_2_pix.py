import keras
from keras import Input, Model
from keras.src.layers import Conv2D, Conv2DTranspose, LeakyReLU, Concatenate
from keras.src.ops import ones_like, zeros_like
from tensorflow import GradientTape
import tensorflow as tf

loss_object = keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)

def build_generator():
    inputs = Input(shape=[256, 256, 1])

    down1 = Conv2D(64, (4, 4), strides=2, padding="same", activation="relu")(inputs)
    down2 = Conv2D(128, (4, 4), strides=2, padding="same", activation="relu")(down1)
    down3 = Conv2D(256, (4, 4), strides=2, padding="same", activation="relu")(down2)

    up1 = Conv2DTranspose(128, (4, 4), strides=2, padding="same", activation="relu")(down3)
    up2 = Conv2DTranspose(64, (4, 4), strides=2, padding="same", activation="relu")(up1)

    outputs = Conv2DTranspose(3, (4, 4), strides=2, padding="same", activation="tanh")(up2)

    return Model(inputs, outputs)

def build_discriminator():
    sketch_input = Input(shape=[256, 256, 1])
    original_input = Input(shape=[256, 256, 3])

    x = Concatenate()([sketch_input, original_input])
    x = Conv2D(64, (4, 4), strides=2, padding="same")(x)
    x = LeakyReLU()(x)

    x = Conv2D(128, (4, 4), strides=2, padding="same")(x)
    x = LeakyReLU()(x)

    x = Conv2D(1, (4, 4), padding="same")(x)

    return Model([sketch_input, original_input], x)

def generator_loss(fake_output):
    return loss_object(ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = loss_object(ones_like(real_output), real_output)
    fake_loss = loss_object(zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

@tf.function
def train_step(sketches, originals, generator, discriminator):
    with GradientTape() as gen_tape, GradientTape() as disc_tape:
        generated_images = generator(sketches, training=True)
        real_output = discriminator([sketches, originals], training=True)
        fake_output = discriminator([sketches, generated_images], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
