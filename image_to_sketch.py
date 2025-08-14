import cv2
import matplotlib.image
import os

def burnV2(image, mask):
    return 255 - cv2.divide(255 - image, 255 - mask, scale=256)

def dodgeV2(image, mask):
    return cv2.divide(image, 255 - mask, scale=256)

def regenerate_sketch(image):
    """
    Generate a sketch from a single image array (compatible with augmentation functions).
    
    Args:
        image: Input image as numpy array (BGR format from cv2.imread)
    
    Returns:
        Sketch image as numpy array (BGR format)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image
    
    # Apply the basic sketch generation algorithm
    img_gray = cv2.convertScaleAbs(img_gray, alpha=0.6, beta=30)
    img_gray_inv = 255 - img_gray
    img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(9, 9), sigmaX=0, sigmaY=0)
    img_blend = dodgeV2(img_gray, img_blur)
    
    # Threshold to simplify lines - only most important contours
    _, img_blend = cv2.threshold(img_blend, 240, 255, cv2.THRESH_BINARY)
    
    # Convert back to 3-channel for consistency with other augmentation functions
    if len(image.shape) == 3:
        img_blend = cv2.cvtColor(img_blend, cv2.COLOR_GRAY2BGR)
    
    return img_blend

def process_folder_to_sketches(input_folder, output_folder):
    """
    Process all images in input folder and convert them to sketches
    """
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder does not exist: {input_folder}")
        return
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image formats
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(supported_formats)]
    
    if not image_files:
        print(f"No supported image files found in: {input_folder}")
        return
    
    print(f"Found {len(image_files)} files to process")
    
    processed = 0
    failed = 0
    
    for filename in image_files:
        try:
            input_path = os.path.join(input_folder, filename)
            
            # Load image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Cannot load image: {input_path}")
                failed += 1
                continue
            
            # Generate sketch
            sketch = regenerate_sketch(image)
            
            # Create output filename
            name_without_ext = os.path.splitext(filename)[0]
            output_filename = f"{name_without_ext}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            
            # Save sketch
            cv2.imwrite(output_path, sketch)
            print(f"Generated sketch: {output_path}")
            processed += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            failed += 1
    
    print(f"Processing completed. Successful: {processed}, Failed: {failed}")

# Configuration
if __name__ == "__main__":
    # Set your input and output folders here
    input_folder = "./dataset/faces_custom/train/original"
    output_folder = "./dataset/faces_custom/train/sketch"
    
    # Run processing
    process_folder_to_sketches(input_folder, output_folder)
