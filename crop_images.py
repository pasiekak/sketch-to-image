import os
import cv2
import numpy as np
from config import IMG_SIZE  # lub wpisz np. IMG_SIZE = 256

def load_and_crop_image(path, size=IMG_SIZE):
    target_h, target_w = (size, size) if isinstance(size, int) else size

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape
    scale = max(target_w / w, target_h / h)
    resized_w = max(int(w * scale), target_w)
    resized_h = max(int(h * scale), target_h)

    img_resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    x_start = max((resized_w - target_w) // 2, 0)
    y_start = max((resized_h - target_h) // 2, 0)
    img_cropped = img_resized[y_start:y_start + target_h, x_start:x_start + target_w]

    return cv2.cvtColor(img_cropped.astype(np.uint8), cv2.COLOR_RGB2BGR)

def crop_all_images(input_folder):
    output_folder = input_folder.rstrip('/\\') + '_256'
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Cropping {len(image_files)} images from {input_folder} → {output_folder}")

    for i, fname in enumerate(image_files, 1):
        input_path = os.path.join(input_folder, fname)
        output_path = os.path.join(output_folder, fname)

        try:
            cropped_img = load_and_crop_image(input_path)
            cv2.imwrite(output_path, cropped_img)
        except Exception as e:
            print(f"Failed to process {fname}: {e}")

        if i % 50 == 0 or i == len(image_files):
            print(f"Processed {i}/{len(image_files)}")

    print("Done.")

if __name__ == '__main__':
    input_dir = './dataset/train/original512'
    crop_all_images(input_dir)
