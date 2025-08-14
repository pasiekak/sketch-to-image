import os
import cv2
import numpy as np
from image_to_sketch import regenerate_sketch

# ========================
# Augmentation definitions
# ========================
def make_lines_thicker(image):
    """Make sketch lines thicker using erode operation (for white background, black lines)"""
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def random_erase_fragments(image):
    """Randomly erase fragments of the sketch"""
    result = image.copy()
    h, w = result.shape[:2]
    
    # Number of fragments to erase (random between 3-8)
    num_fragments = np.random.randint(5, 13)
    
    for _ in range(num_fragments):
        # Random fragment size (5-20% of image dimensions)
        frag_w = np.random.randint(int(w * 0.05), int(w * 0.2))
        frag_h = np.random.randint(int(h * 0.05), int(h * 0.2))
        
        # Random position
        x = np.random.randint(0, w - frag_w)
        y = np.random.randint(0, h - frag_h)
        
        # Erase fragment by setting it to white (255, 255, 255)
        if len(result.shape) == 3:
            result[y:y+frag_h, x:x+frag_w] = (255, 255, 255)
        else:
            result[y:y+frag_h, x:x+frag_w] = 255
    
    return result

def add_noise_to_lines(image):
    """Add noise to sketch lines to simulate hand-drawn imperfections"""
    result = image.copy()
    
    # Find line pixels (non-white pixels)
    if len(result.shape) == 3:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    else:
        gray = result.copy()
    
    line_pixels = gray < 200  # Pixels that are not white
    
    # Add random noise to line pixels
    noise = np.random.randint(-30, 30, size=gray.shape)
    noisy_gray = gray.astype(np.int16) + (noise * line_pixels)
    noisy_gray = np.clip(noisy_gray, 0, 255).astype(np.uint8)
    
    if len(result.shape) == 3:
        result = cv2.cvtColor(noisy_gray, cv2.COLOR_GRAY2BGR)
    else:
        result = noisy_gray
    
    return result

def vary_line_intensity(image):
    """Vary the intensity of sketch lines"""
    result = image.copy()
    
    if len(result.shape) == 3:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    else:
        gray = result.copy()
    
    # Create intensity variation (darker or lighter lines)
    intensity_factor = np.random.uniform(0.5, 1.5)
    
    # Apply only to non-white pixels
    line_mask = gray < 200
    gray_float = gray.astype(np.float32)
    gray_float[line_mask] = gray_float[line_mask] * intensity_factor
    gray_float = np.clip(gray_float, 0, 255)
    
    result_gray = gray_float.astype(np.uint8)
    
    if len(result.shape) == 3:
        result = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)
    else:
        result = result_gray
    
    return result

def partial_line_breaks(image):
    """Create small breaks in lines to simulate incomplete sketching"""
    result = image.copy()
    h, w = result.shape[:2]
    
    if len(result.shape) == 3:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    else:
        gray = result.copy()
    
    # Find line pixels
    line_pixels = np.where(gray < 200)
    
    if len(line_pixels[0]) > 0:
        # Randomly select some line pixels to erase
        num_breaks = min(len(line_pixels[0]) // 20, 50)  # Break ~5% of line pixels
        break_indices = np.random.choice(len(line_pixels[0]), num_breaks, replace=False)
        
        for idx in break_indices:
            y, x = line_pixels[0][idx], line_pixels[1][idx]
            # Create small white spots (line breaks)
            if len(result.shape) == 3:
                result[max(0,y-1):min(h,y+2), max(0,x-1):min(w,x+2)] = (255, 255, 255)
            else:
                result[max(0,y-1):min(h,y+2), max(0,x-1):min(w,x+2)] = 255
    
    return result

def sketch_rotation(image, max_angle=5):
    """Small rotations to simulate natural hand movement"""
    h, w = image.shape[:2]
    angle = np.random.uniform(-max_angle, max_angle)
    
    # Rotation matrix
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    
    # Rotate with white background
    rotated = cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))
    return rotated

def gaussian_blur_light(image):
    """Apply light gaussian blur to simulate slight smudging"""
    kernel_size = np.random.choice([3, 5])
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def random_line_thickness_variation(image):
    """Create lines with varying thickness in the same sketch"""
    result = image.copy()
    
    if len(result.shape) == 3:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    else:
        gray = result.copy()
    
    # Find line pixels
    line_pixels = gray < 200
    
    # Create random regions with different thickness
    h, w = gray.shape
    num_regions = np.random.randint(3, 6)
    
    for _ in range(num_regions):
        # Random region
        region_w = np.random.randint(w//4, w//2)
        region_h = np.random.randint(h//4, h//2)
        x = np.random.randint(0, w - region_w)
        y = np.random.randint(0, h - region_h)
        
        region_mask = np.zeros_like(gray, dtype=bool)
        region_mask[y:y+region_h, x:x+region_w] = True
        
        # Apply thickness change only to line pixels in this region
        region_lines = line_pixels & region_mask
        
        if np.any(region_lines):
            # Choose thick or thin randomly
            if np.random.choice([True, False]):
                # Make thicker (erode)
                kernel = np.ones((3, 3), np.uint8)
                region_result = cv2.erode(result[y:y+region_h, x:x+region_w], kernel, iterations=1)
            else:
                # Make thinner (dilate)
                kernel = np.ones((3, 3), np.uint8)
                region_result = cv2.dilate(result[y:y+region_h, x:x+region_w], kernel, iterations=1)
            
            result[y:y+region_h, x:x+region_w] = region_result
    
    return result

SKETCH_AUGMENTATIONS = {
    'thick': {'function': make_lines_thicker, 'folder': 'thick'},
    'resketch': {'function': regenerate_sketch, 'folder': 'resketch', 'use_original': True},
    'erased': {'function': random_erase_fragments, 'folder': 'erased'},
    'noisy': {'function': add_noise_to_lines, 'folder': 'noisy'},
    'varied_intensity': {'function': vary_line_intensity, 'folder': 'varied_intensity'},
    'broken_lines': {'function': partial_line_breaks, 'folder': 'broken_lines'},
    'rotated': {'function': lambda img: sketch_rotation(img, np.random.uniform(2, 8)), 'folder': 'rotated'},
    'blurred': {'function': gaussian_blur_light, 'folder': 'blurred'},
    'varied_thickness': {'function': random_line_thickness_variation, 'folder': 'varied_thickness'},
}

# ========================
# Augmentation pipeline
# ========================
def augment_sketches(input_folder, original_folder):
    """
    Augments sketches by creating versions with thicker and thinner lines, and regenerated sketches.
    
    Args:
        input_folder: Path to folder containing sketch images
        original_folder: Path to folder containing original images (required for resketch augmentation)
    """
    # Create augmented folder structure
    parent_dir = os.path.dirname(input_folder)
    augmented_dir = os.path.join(parent_dir, 'augmented')
    
    # Create subfolders for different augmentation types
    for aug_name, aug_config in SKETCH_AUGMENTATIONS.items():
        aug_folder = os.path.join(augmented_dir, aug_config['folder'])
        os.makedirs(aug_folder, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for i, fname in enumerate(image_files, 1):
        # Skip already augmented files (those with numeric suffixes like -1, -2, etc.)
        name_without_ext, ext = os.path.splitext(fname)
        if '-' in name_without_ext and name_without_ext.split('-')[-1].isdigit():
            continue
            
        path = os.path.join(input_folder, fname)
        image = cv2.imread(path)
        if image is None:
            continue

        # Get name and extension
        name_without_ext, ext = os.path.splitext(fname)
        
        # Apply all augmentations
        for aug_name, aug_config in SKETCH_AUGMENTATIONS.items():
            aug_folder = aug_config['folder']
            aug_function = aug_config['function']
            use_original = aug_config.get('use_original', False)
            
            if use_original:
                # Handle resketch augmentation - use original image
                original_path = os.path.join(original_folder, fname)
                if os.path.exists(original_path):
                    original_image = cv2.imread(original_path)
                    if original_image is not None:
                        aug_image = aug_function(original_image)
                        output_path = os.path.join(augmented_dir, aug_folder, fname)
                        cv2.imwrite(output_path, aug_image)
                    else:
                        print(f"Failed to load original image for {fname}")
                else:
                    print(f"Original image not found for {fname}")
            else:
                # Handle regular augmentations (thick/thin) - use sketch image
                aug_image = aug_function(image)
                aug_path = os.path.join(augmented_dir, aug_folder, fname)
                cv2.imwrite(aug_path, aug_image)

        # Count only original files (without numeric suffixes) for progress
        original_files = [f for f in image_files 
                         if not ('-' in os.path.splitext(f)[0] and os.path.splitext(f)[0].split('-')[-1].isdigit())]
        
        if i % 50 == 0 or i == len(original_files):
            print(f"Processed {i}/{len(original_files)} original images")

    print("Sketch augmentation complete.")

if __name__ == '__main__':
    # Usage example
    sketch_dir = "./dataset/faces_all/train/sketch"  # Folder with sketches
    original_dir = "./dataset/faces_all/train/original"  # Folder with original images
    
    # Run all augmentations (thick, thin, resketch)
    augment_sketches(sketch_dir, original_dir)