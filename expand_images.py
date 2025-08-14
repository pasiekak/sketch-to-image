import os
import cv2
import numpy as np
from config import IMG_SIZE

def pad_image_to_size(path, size=IMG_SIZE):
    """Expands canvas to target size, centers image and fills empty space with white background"""
    target_h, target_w = (size, size) if isinstance(size, int) else size

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    
    h, w, c = img.shape
    
    # If image is larger than target size, scale it down proportionally
    if h > target_h or w > target_w:
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w
    
    # Create white canvas with target size
    result = np.full((target_h, target_w, c), 255, dtype=np.uint8)
    
    # Calculate position to center the image
    start_y = (target_h - h) // 2
    start_x = (target_w - w) // 2
    
    # Place image in the center of white canvas
    result[start_y:start_y + h, start_x:start_x + w] = img
    
    return result

def pad_all_images(input_folder, output_folder=None):
    """Processes all images in the folder"""
    if output_folder is None:
        output_folder = input_folder.rstrip('/\\') + '_padded_256'
    
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Padding {len(image_files)} images from {input_folder} → {output_folder}")

    for i, fname in enumerate(image_files, 1):
        input_path = os.path.join(input_folder, fname)
        output_path = os.path.join(output_folder, fname)

        try:
            padded_img = pad_image_to_size(input_path)
            cv2.imwrite(output_path, padded_img)
        except Exception as e:
            print(f"Failed to process {fname}: {e}")

        if i % 50 == 0 or i == len(image_files):
            print(f"Processed {i}/{len(image_files)}")

    print("Done.")

if __name__ == '__main__':
    # Configuration - modify these paths as needed
    input_dir = './dataset/boots/original'
    output_dir = './dataset/boots/original_256'
    
    pad_all_images(input_dir, output_dir)