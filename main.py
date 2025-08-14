from config import BATCH_SIZE, EPOCHS, USE_GPU
from model import create_discriminatior70x70, create_gan, create_generator, train
from utils import load_dataset, preprocess_data

import os

# Set CUDA_VISIBLE_DEVICES before importing tensorflow
if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)
os.chdir(os.path.dirname(__file__))

dataset = load_dataset()
image_shape = dataset[0].shape[1:]
dataset = preprocess_data(dataset)

# Create models
print("Creating discriminator...")
discriminator_model = create_discriminatior70x70(image_shape)

print("Creating generator...")
generator_model = create_generator(image_shape)

print("Creating GAN...")
gan_model = create_gan(generator_model, discriminator_model, image_shape)

print("GAN SUMMARY")
gan_model.summary()

print("Starting training...")
train(discriminator_model, generator_model, gan_model, dataset, n_epochs=EPOCHS, n_batch=BATCH_SIZE)