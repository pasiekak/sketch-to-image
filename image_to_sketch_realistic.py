import cv2
import numpy as np
import matplotlib.image
import os

def burnV2(image, mask):
    return 255 - cv2.divide(255 - image, 255 - mask, scale=256)

def dodgeV2(image, mask):
    return cv2.divide(image, 255 - mask, scale=256)

def convert_image_to_sketch(image, k_size=21):
    """
    Convert a single image to pencil sketch using simplified algorithm.
    
    Algorithm steps:
    1. Convert to grayscale
    2. Invert the grayscale image
    3. Apply Gaussian blur to inverted image
    4. Invert the blurred image
    5. Divide grayscale by inverted blur to create sketch effect
    
    Args:
        image: Input image as numpy array (BGR format from cv2.imread)
        k_size: Kernel size for Gaussian blur (default: 21)
    
    Returns:
        Sketch image as numpy array (grayscale)
    """
    # Step 1: Convert to grayscale
    if len(image.shape) == 3:
        grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grey_img = image
    
    # Step 2: Invert the grayscale image
    invert_img = cv2.bitwise_not(grey_img)
    
    # Step 3: Apply Gaussian blur to inverted image
    blur_img = cv2.GaussianBlur(invert_img, (k_size, k_size), 0)
    
    # Step 4: Invert the blurred image
    invblur_img = cv2.bitwise_not(blur_img)
    
    # Step 5: Create sketch by dividing grayscale by inverted blur
    sketch_img = cv2.divide(grey_img, invblur_img, scale=256.0)
    
    return sketch_img

def process_folder_to_sketches(image_folder, sketch_folder, k_size=21):
    """
    Process all images in input folder and convert them to sketches using convert_image_to_sketch function.
    
    Args:
        image_folder: Path to folder containing original images
        sketch_folder: Path to folder where sketches will be saved
        k_size: Kernel size for Gaussian blur (default: 21)
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(sketch_folder):
        os.makedirs(sketch_folder)

    # Get list of image files
    img_names = os.listdir(image_folder)
    
    print(f"Processing {len(img_names)} images from {image_folder} with kernel size {k_size}")
    processed = 0
    failed = 0

    for img in img_names:
        try:
            img_path = os.path.join(image_folder, img)
            sketch_path = os.path.join(sketch_folder, img)

            # Load image
            img_rgb = cv2.imread(img_path)
            if img_rgb is None:
                print(f"Cannot load image: {img_path}")
                failed += 1
                continue

            # Convert to sketch using the dedicated function with specified kernel size
            img_sketch = convert_image_to_sketch(img_rgb, k_size)

            # Save sketch using cv2.imwrite for consistency with original algorithm
            cv2.imwrite(sketch_path, img_sketch)
            print(f"Generated sketch: {sketch_path}")
            processed += 1
            
        except Exception as e:
            print(f"Error processing {img}: {str(e)}")
            failed += 1
    
    print(f"Processing completed for {image_folder}. Successful: {processed}, Failed: {failed}")

if __name__ == "__main__":
    # Configuration paths
    input_folder = './dataset/faces_men/test/original'
    output_folder = './dataset/faces_men/test/sketch'
    
    # Kernel size for Gaussian blur (smaller = more detailed, larger = smoother)
    # Common values: 7, 15, 21, 31
    kernel_size = 7
    
    print("Starting sketch conversion process...")
    print(f"Using kernel size: {kernel_size}")
    
    # Process images from input to output folder
    if os.path.exists(input_folder):
        process_folder_to_sketches(input_folder, output_folder, kernel_size)
    else:
        print(f"Input folder not found: {input_folder}")
    
    print("Processing completed!")