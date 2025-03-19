import cv2
import matplotlib.image
import os

from constants import TESTING_ORIGINAL_IMAGES_PATH, TESTING_SKETCH_IMAGES_PATH

training_original = './datasets/train/original'
training_sketch = './datasets/train/sketch'

test_original = './datasets/test/original'
test_sketch = './datasets/test/sketch'

def burnV2(image, mask):
    return 255 - cv2.divide(255 - image, 255 - mask, scale=256)


def dodgeV2(image, mask):
    return cv2.divide(image, 255 - mask, scale=256)


def img_to_sketch(image_folder, sketch_folder):
    if not os.path.exists(sketch_folder):
        os.makedirs(sketch_folder)

    img_names = os.listdir(image_folder)

    for img in img_names:
        img_path = os.path.join(image_folder, img)
        sketch_path = os.path.join(sketch_folder, img)


        img_rgb = cv2.imread(img_path)
        if img_rgb is None:
            continue

        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img_gray_inv = 255 - img_gray
        img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21), sigmaX=0, sigmaY=0)
        img_blend = dodgeV2(img_gray, img_blur)

        matplotlib.image.imsave(sketch_path, img_blend, cmap='gray')

for folder in os.listdir(training_original):
    image_folder = os.path.join(training_original, folder)
    sketch_folder = os.path.join(training_sketch, folder)

    if os.path.isdir(image_folder):
        img_to_sketch(image_folder, sketch_folder)

# TESTOWY ZBIÓR
img_to_sketch(TESTING_ORIGINAL_IMAGES_PATH, TESTING_SKETCH_IMAGES_PATH)

