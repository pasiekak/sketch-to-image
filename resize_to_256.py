import cv2
import os

def resize_to_256(image_path, output_path):
    """
    Resize image to 256x256 pixels
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot load image: {image_path}")
            return False
        
        # Resize image to 256x256
        resized_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        
        # Save resized image
        cv2.imwrite(output_path, resized_image)
        print(f"Resized and saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def process_folder(input_folder, output_folder):
    """
    Process all images in the input folder and resize them to 256x256
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
        input_path = os.path.join(input_folder, filename)
        
        # Keep the same filename but change extension to .jpg
        name_without_ext = os.path.splitext(filename)[0]
        output_filename = f"{name_without_ext}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        
        if resize_to_256(input_path, output_path):
            processed += 1
        else:
            failed += 1
    
    print(f"Processing completed. Successful: {processed}, Failed: {failed}")

if __name__ == "__main__":
    input_folder = "./dataset/faces_augmented/train/sketch"
    output_folder = "./dataset/faces_augmented/train/sketch"
    
    process_folder(input_folder, output_folder)
