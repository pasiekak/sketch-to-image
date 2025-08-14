from config import TRAINING_DATASET_PATH, MAX_TRAINING_SAMPLES, RANDOM_SEED, USE_AUGMENTATION
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
import glob
import random

def preprocess_data(data):
    X1, X2 = data[0], data[1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]

def load_images(dataset_base_path, size=(256, 256)):
    """Load paired sketch and original images (including augmented variants)"""
    src_list, tar_list = list(), list()

    sketch_folder = os.path.join(dataset_base_path, 'sketch')
    original_folder = os.path.join(dataset_base_path, 'original')
    augmented_root = os.path.join(dataset_base_path, 'augmented')

    loaded_pairs = 0

    # Collect all sketch sources: original sketches + augmented sketches
    sketch_sources = []
    
    # 1. Add original sketches
    if os.path.exists(sketch_folder):
        sketch_files = [f for f in os.listdir(sketch_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"DATASET: Found {len(sketch_files)} original sketches in {sketch_folder}")
        for sketch_file in sketch_files:
            sketch_sources.append(('original_sketch', os.path.join(sketch_folder, sketch_file), sketch_file))
    else:
        print(f"DATASET: Original sketch folder not found: {sketch_folder}")
    
    # 2. Add augmented sketches (only if USE_AUGMENTATION is True)
    if USE_AUGMENTATION and os.path.isdir(augmented_root):
        print(f"DATASET: Checking augmented folder: {augmented_root}")
        aug_folders = [f for f in os.listdir(augmented_root) if os.path.isdir(os.path.join(augmented_root, f))]
        print(f"DATASET: Found augmentation types: {aug_folders}")
        
        for aug_type in aug_folders:
            aug_path = os.path.join(augmented_root, aug_type)
            aug_files = [f for f in os.listdir(aug_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"DATASET: Found {len(aug_files)} {aug_type} augmented sketches")
            for aug_file in aug_files:
                sketch_sources.append((f'aug_{aug_type}', os.path.join(aug_path, aug_file), aug_file))
    elif not USE_AUGMENTATION:
        print("DATASET: Augmentation disabled - using only original sketches")
    else:
        print(f"DATASET: Augmented folder not found: {augmented_root}")

    print(f"DATASET: Total sketch sources collected: {len(sketch_sources)}")
    
    # Show breakdown by type
    type_counts = {}
    for source_type, _, _ in sketch_sources:
        type_counts[source_type] = type_counts.get(source_type, 0) + 1
    
    for source_type, count in type_counts.items():
        print(f"DATASET: {source_type}: {count} sketches")

    # Shuffle all sketch sources
    random.seed(RANDOM_SEED)
    random.shuffle(sketch_sources)

    for source_type, sketch_path, filename in sketch_sources:
        if loaded_pairs >= MAX_TRAINING_SAMPLES:
            break

        if loaded_pairs % 100 == 0:
            print(f"DATASET: Loaded {loaded_pairs} image pairs")

        # For each sketch (original or augmented), find matching original image
        base_name = os.path.splitext(filename)[0]
        original_path = os.path.join(original_folder, filename)
        
        # Try different extensions if exact match not found
        if not os.path.exists(original_path):
            matching_files = glob.glob(os.path.join(original_folder, base_name + '.*'))
            if matching_files:
                original_path = matching_files[0]
            else:
                print(f"No matching original found for {filename} (from {source_type})")
                continue

        try:
            sketch = load_img(sketch_path, target_size=size)
            sketch = img_to_array(sketch)

            original = load_img(original_path, target_size=size)
            original = img_to_array(original)

            src_list.append(sketch)
            tar_list.append(original)
            loaded_pairs += 1

        except Exception as e:
            print(f"Error loading pair {filename} (from {source_type}) & {original_path}: {e}")

    print(f"DATASET: Loaded {loaded_pairs} image pairs")
    return [np.array(src_list), np.array(tar_list)]

def load_dataset():
    dataset_base_path = os.path.join(os.getcwd(), TRAINING_DATASET_PATH)

    if not os.path.isdir(dataset_base_path):
        raise ValueError(f"Dataset path not found: {dataset_base_path}")

    sketches, originals = load_images(dataset_base_path)
    return sketches, originals
