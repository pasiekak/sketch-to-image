import cv2
import numpy as np

from constants import IMG_SIZE


def load_image(sketch_path, original_path):
    sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
    sketch = cv2.resize(sketch, (IMG_SIZE, IMG_SIZE))
    sketch = sketch / 255.0

    original = cv2.imread(original_path)
    original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    original = original / 255.0

    return sketch[..., np.newaxis], original