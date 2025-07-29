import os
import numpy as np
from keras._tf_keras.keras.utils import load_img, img_to_array
from config import TRAINING_ORIGINAL_IMAGES_PATH, TRAINING_SKETCH_IMAGES_PATH, TRAINING_CATEGORY

def load_images(sketch_folder, original_folder, size=(256,256)):
    """Load paired sketch and original images from separate folders"""
    src_list, tar_list = list(), list()
    
    # Get all sketch files
    sketch_files = [f for f in os.listdir(sketch_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for sketch_filename in sketch_files:
        sketch_path = os.path.join(sketch_folder, sketch_filename)
        
        # Extract base name (part before the dash) to find corresponding original
        # e.g., "n02691156_10151-1.png" -> "n02691156_10151"
        if '-' in sketch_filename:
            base_name = sketch_filename.split('-')[0]
            # Find original file with the same base name
            original_filename = None
            for ext in ['.jpg', '.jpeg', '.png']:
                potential_original = base_name + ext
                original_path = os.path.join(original_folder, potential_original)
                if os.path.exists(original_path):
                    original_filename = potential_original
                    break
            
            if original_filename:
                # Load sketch
                sketch = load_img(sketch_path, target_size=size)
                sketch = img_to_array(sketch)
                
                # Load original
                original = load_img(original_path, target_size=size)
                original = img_to_array(original)
                
                src_list.append(sketch)
                tar_list.append(original)
        else:
            # Fallback for old naming convention (1:1 pairing)
            original_path = os.path.join(original_folder, sketch_filename)
            if os.path.exists(original_path):
                # Load sketch
                sketch = load_img(sketch_path, target_size=size)
                sketch = img_to_array(sketch)
                
                # Load original
                original = load_img(original_path, target_size=size)
                original = img_to_array(original)
                
                src_list.append(sketch)
                tar_list.append(original)
    
    return [np.array(src_list), np.array(tar_list)]

def load_dataset(category=None, max_samples=None, random_seed=42):
    """Load dataset from a specific category or from config"""
    if category is None:
        category = TRAINING_CATEGORY
    
    # Check if user wants to load all categories
    if category.lower() == 'all':
        return load_dataset_all_categories()
    
    # Get base directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sketch_base = os.path.join(base_dir, TRAINING_SKETCH_IMAGES_PATH)
    original_base = os.path.join(base_dir, TRAINING_ORIGINAL_IMAGES_PATH)
    
    # Load specific category
    sketch_folder = os.path.join(sketch_base, category)
    original_folder = os.path.join(original_base, category)
    
    if os.path.isdir(sketch_folder) and os.path.isdir(original_folder):
        print(f"Loading category: {category}")
        sketches, originals = load_images(sketch_folder, original_folder)
        
        # Limit samples if requested
        if max_samples and max_samples < len(sketches):
            np.random.seed(random_seed)
            indices = np.random.choice(len(sketches), max_samples, replace=False)
            sketches = sketches[indices]
            originals = originals[indices]
            print(f"Limited to {max_samples} random samples (from {len(indices)} total available)")
        
        print(f"Loaded {len(sketches)} image pairs from {category}")
        print(f"Sketch shape: {sketches.shape}")
        print(f"Original shape: {originals.shape}")
        
        return sketches, originals
    else:
        raise ValueError(f"Category '{category}' not found in dataset")

def load_dataset_all_categories():
    """Load complete dataset from all categories (memory intensive)"""
    all_sketches = []
    all_originals = []
    
    # Get base directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sketch_base = os.path.join(base_dir, TRAINING_SKETCH_IMAGES_PATH)
    original_base = os.path.join(base_dir, TRAINING_ORIGINAL_IMAGES_PATH)
    
    # Get all categories
    categories = os.listdir(sketch_base)
    
    for category in categories:
        sketch_folder = os.path.join(sketch_base, category)
        original_folder = os.path.join(original_base, category)
        
        if os.path.isdir(sketch_folder) and os.path.isdir(original_folder):
            print(f"Loading category: {category}")
            sketches, originals = load_images(sketch_folder, original_folder)
            
            all_sketches.extend(sketches)
            all_originals.extend(originals)
    
    # Convert to numpy arrays
    train_sketches = np.array(all_sketches)
    train_originals = np.array(all_originals)
    
    print(f"Loaded {len(train_sketches)} image pairs")
    print(f"Sketch shape: {train_sketches.shape}")
    print(f"Original shape: {train_originals.shape}")
    
    return train_sketches, train_originals

def get_available_categories():
    """Get list of available categories in the dataset"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sketch_base = os.path.join(base_dir, TRAINING_SKETCH_IMAGES_PATH)
    
    if not os.path.exists(sketch_base):
        return []
    
    categories = [d for d in os.listdir(sketch_base) 
                 if os.path.isdir(os.path.join(sketch_base, d))]
    return sorted(categories)

def print_available_categories():
    """Print all available categories"""
    categories = get_available_categories()
    print(f"Available categories ({len(categories)}):")
    for i, category in enumerate(categories, 1):
        print(f"  {i:2d}. {category}")
    print(f"\nTo change category, modify TRAINING_CATEGORY in config.py")
    print(f"Current category: {TRAINING_CATEGORY}")
    