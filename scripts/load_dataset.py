import os

import numpy as np
import tensorflow as tf

from constants import BUFFER_SIZE, BATCH_SIZE, TRAINING_ORIGINAL_IMAGES_PATH, TRAINING_SKETCH_IMAGES_PATH, \
    TESTING_SKETCH_IMAGES_PATH, TESTING_ORIGINAL_IMAGES_PATH
from scripts.load_image import load_image

def load_dataset():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    train_sketch_folder = os.path.normpath(os.path.join(base_dir, TRAINING_SKETCH_IMAGES_PATH))
    train_original_folder = os.path.normpath(os.path.join(base_dir, TRAINING_ORIGINAL_IMAGES_PATH))
    test_original_folder = os.path.normpath(os.path.join(base_dir, TESTING_ORIGINAL_IMAGES_PATH))
    test_sketch_folder = os.path.normpath(os.path.join(base_dir, TESTING_SKETCH_IMAGES_PATH))

    train_sketches = sorted(os.listdir(train_sketch_folder))
    train_originals = sorted(os.listdir(train_original_folder))

    test_sketches = sorted(os.listdir(test_sketch_folder))
    test_originals = sorted(os.listdir(test_original_folder))

    train_dataset = []
    for sk, og in zip(train_sketches, train_originals):
        sk_path = os.path.join(train_sketch_folder, sk)
        og_path = os.path.join(train_original_folder, og)
        for filename in os.listdir(sk_path):
            sk_filepath = os.path.join(sk_path, filename)
            og_filepath = os.path.join(og_path, filename)

            train_dataset.append(load_image(sk_filepath, og_filepath))

    test_dataset = []
    for sk, og in zip(test_sketches, test_originals):
        sk_path = os.path.join(test_sketch_folder, sk)
        og_path = os.path.join(test_original_folder, og)

        test_dataset.append(load_image(sk_path, og_path))

    train_sketches, train_originals = zip(*train_dataset)
    test_sketches, test_originals = zip(*test_dataset)

    train_sketches = np.array(train_sketches, dtype=np.float32)
    train_originals = np.array(train_originals, dtype=np.float32)

    test_sketches = np.array(test_sketches, dtype=np.float32)
    test_originals = np.array(test_originals, dtype=np.float32)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_sketches, train_originals)).shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_sketches, test_originals)).batch(BATCH_SIZE)

    return train_dataset, test_dataset