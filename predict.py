import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import glob

from config import IMG_SIZE

# ============================
# Configuration
# ============================

# Model experiment name - choose which experiment to process
MODEL_NAME = '#FACES_WOMEN_S20000_E100_B1_LR0.0001_B10.5_B20.999_LRA0.2_DO0.25_GW1_L1W200_MSI1_AUG0'

# Base models directory
MODELS_BASE_DIR = './models'

# Experiment directory
EXPERIMENT_DIR = os.path.join(MODELS_BASE_DIR, MODEL_NAME)

# ============================
# Core prediction functions
# ============================

def generate_single_image(model, sketch_array):
    """
    Generate a single image from a sketch using the given model.
    
    Args:
        model: Loaded generator model
        sketch_array: Preprocessed sketch array (should be normalized to [-1, 1])
    
    Returns:
        generated_image: Generated image array (rescaled to [0, 1])
    """
    if len(sketch_array.shape) == 3:
        sketch_batch = np.expand_dims(sketch_array, axis=0)
    else:
        sketch_batch = sketch_array
    
    generated = model.predict(sketch_batch, verbose=0)[0]
    generated = (generated + 1) / 2.0  # Rescale to [0,1] for saving
    
    return generated

def preprocess_sketch(sketch_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Load and preprocess a sketch image for model input.
    
    Args:
        sketch_path: Path to the sketch image
        target_size: Target size for the image (width, height)
    
    Returns:
        sketch_array: Original sketch array
        normalized_sketch: Normalized sketch array for model input [-1, 1]
    """
    sketch = load_img(sketch_path, target_size=target_size)
    sketch_array = img_to_array(sketch)
    normalized_sketch = (sketch_array - 127.5) / 127.5  # Normalize to [-1, 1]
    
    return sketch_array, normalized_sketch

def generate_image_from_path(model_path, sketch_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Generate a single image from a sketch path using a model path.
    This is a convenience function that loads the model, preprocesses the sketch,
    and generates the image in one call.
    
    Args:
        model_path: Path to the trained generator model
        sketch_path: Path to the sketch image
        target_size: Target size for the image (width, height)
    
    Returns:
        tuple: (original_sketch_array, generated_image_array)
    """
    model = load_model(model_path, compile=False)
    sketch_array, normalized_sketch = preprocess_sketch(sketch_path, target_size)
    generated = generate_single_image(model, normalized_sketch)
    
    return sketch_array, generated

def create_models_comparison_summary(experiment_dir, num_samples=None):
    """
    Create a comparison summary showing original, sketch, and predictions from all models
    in the experiment. Layout: original | sketch | model1_pred | model2_pred | model3_pred | ...
    
    Args:
        experiment_dir: Path to the experiment directory
        num_samples: Number of sample rows to include in comparison (if None, uses all available)
    
    Returns:
        str: Path to the created comparison summary image
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.gridspec import GridSpec
    
    # Find all prediction directories
    predictions_dir = os.path.join(experiment_dir, "predictions")
    
    if not os.path.exists(predictions_dir):
        print(f"No predictions directory found in {experiment_dir}")
        return None
    
    # Find all unique model names from prediction files
    all_files = [f for f in os.listdir(predictions_dir) if f.endswith('_P.jpg')]
    model_names = set()
    for file in all_files:
        # Extract model name from filename: basename_modelname_P.jpg
        parts = file.rsplit('_', 2)  # Split from right: ['basename', 'modelname', 'P.jpg']
        if len(parts) >= 3:
            model_names.add(parts[1])  # modelname is the second-to-last part
    
    model_names = sorted(list(model_names))
    
    if len(model_names) < 2:
        print(f"Need at least 2 models for comparison, found {len(model_names)}")
        return None
    
    print(f"Creating comparison summary for {len(model_names)} models...")
    for i, model_name in enumerate(model_names, 1):
        print(f"  {i}. {model_name}")
    
    # Find available samples - get all base names (without model suffix)
    base_names = set()
    for file in all_files:
        parts = file.rsplit('_', 2)
        if len(parts) >= 3:
            base_names.add(parts[0])  # basename is the first part
    
    base_names = sorted(list(base_names))
    available_samples = len(base_names)
    
    if available_samples == 0:
        print("No samples found")
        return None
    
    # Use all samples if num_samples is None
    if num_samples is None:
        num_samples = available_samples
    else:
        num_samples = min(num_samples, available_samples)
    
    print(f"Creating comparison with {num_samples} samples...")
    
    # Create figure: rows = num_samples, cols = 2 (original + sketch) + number of models
    num_cols = 2 + len(model_names)
    fig_height = max(2 * num_samples, 6)  # Dynamic height
    fig = plt.figure(figsize=(3 * num_cols, fig_height))
    gs = GridSpec(num_samples, num_cols, figure=fig, hspace=0.15, wspace=0.05)
    
    experiment_name = os.path.basename(experiment_dir)
    fig.suptitle(f'Models Comparison: {experiment_name}', fontsize=16, y=0.98)
    
    for row in range(num_samples):
        base_name = base_names[row]
        
        # Use first model to get original and sketch (they should be identical across models)
        first_model = model_names[0]
        original_file = f"{base_name}_{first_model}_O.jpg"
        sketch_file = f"{base_name}_{first_model}_S.jpg"
        
        original_path = os.path.join(predictions_dir, original_file)
        sketch_path = os.path.join(predictions_dir, sketch_file)
        
        if not os.path.exists(original_path) or not os.path.exists(sketch_path):
            print(f"Missing files for sample {base_name}, skipping...")
            continue
        
        try:
            original_img = mpimg.imread(original_path)
            sketch_img = mpimg.imread(sketch_path)
            
            # Original column
            ax1 = fig.add_subplot(gs[row, 0])
            ax1.imshow(original_img)
            ax1.axis('off')
            if row == 0:
                ax1.set_title('Original', fontsize=14, fontweight='bold', pad=10)
            
            # Sketch column
            ax2 = fig.add_subplot(gs[row, 1])
            ax2.imshow(sketch_img)
            ax2.axis('off')
            if row == 0:
                ax2.set_title('Sketch', fontsize=14, fontweight='bold', pad=10)
            
            # Predictions from each model
            for model_idx, model_name in enumerate(model_names):
                predicted_file = f"{base_name}_{model_name}_P.jpg"
                predicted_path = os.path.join(predictions_dir, predicted_file)
                
                ax = fig.add_subplot(gs[row, 2 + model_idx])
                
                if os.path.exists(predicted_path):
                    try:
                        predicted_img = mpimg.imread(predicted_path)
                        ax.imshow(predicted_img)
                        ax.axis('off')
                        
                        # Add column title only for first row
                        if row == 0:
                            # Extract epoch number from name like "g_model_epoch_020"
                            if 'epoch_' in model_name:
                                epoch_num = model_name.split('epoch_')[1]
                                title = f'Epoch {epoch_num}'
                            else:
                                title = model_name.replace('g_model_', '').replace('g_model', 'Final')
                            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
                    
                    except Exception as e:
                        print(f"Error loading prediction for model {model_name}, sample {base_name}: {str(e)}")
                        ax.axis('off')
                        if row == 0:
                            ax.set_title('Error', fontsize=14, fontweight='bold', pad=10)
                else:
                    print(f"Prediction file not found: {predicted_path}")
                    ax.axis('off')
                    if row == 0:
                        ax.set_title('Missing', fontsize=14, fontweight='bold', pad=10)
        
        except Exception as e:
            print(f"Error processing sample {base_name}: {str(e)}")
            continue
    
    # Save comparison summary
    comparison_path = os.path.join(experiment_dir, "models_comparison_summary.jpg")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Models comparison summary saved: {comparison_path}")
    return comparison_path

def find_all_generator_models(experiment_dir):
    """
    Find all generator model files in a specific experiment directory.
    
    Args:
        experiment_dir: Path to the specific experiment directory
    
    Returns:
        list: List of tuples (experiment_name, model_file_path, output_dir_path)
    """
    generator_models = []
    
    if not os.path.exists(experiment_dir):
        print(f"Experiment directory does not exist: {experiment_dir}")
        return generator_models
    
    experiment_name = os.path.basename(experiment_dir)
    models_dir = os.path.join(experiment_dir, "models")
    g_model_pattern = os.path.join(models_dir, "g_model*.keras")
    g_model_files = glob.glob(g_model_pattern)
    
    # Use single predictions folder for all models
    predictions_dir = os.path.join(experiment_dir, "predictions")
    
    for g_model_file in g_model_files:
        model_filename = os.path.basename(g_model_file)
        model_name = os.path.splitext(model_filename)[0]
        
        generator_models.append((experiment_name, g_model_file, predictions_dir, model_name))
    
    return generator_models

def generate_predictions_for_experiment(experiment_dir, 
                                      sketch_folder=None, 
                                      original_folder=None):
    """
    Generate predictions for all generator models in a specific experiment directory.
    
    Args:
        experiment_dir: Path to the specific experiment directory
        sketch_folder: Folder containing sketch images (if None, will be determined automatically)
        original_folder: Folder containing original images (if None, will be determined automatically)
    """
    generator_models = find_all_generator_models(experiment_dir)
    
    if not generator_models:
        print(f"No generator models found in {experiment_dir}")
        return
    
    experiment_name = os.path.basename(experiment_dir)
    print(f"Processing experiment: {experiment_name}")
    print(f"Found {len(generator_models)} generator models:")
    for i, (exp_name, model_file, output_dir, model_name) in enumerate(generator_models, 1):
        print(f"  {i}. {os.path.basename(model_file)}")
    
    # Determine dataset paths based on experiment name
    if sketch_folder is None or original_folder is None:
        if 'FACES' in experiment_name:
            default_sketch = './dataset/faces_all/test/sketch'
            default_original = './dataset/faces_all/test/original'
        elif 'BOOTS' in experiment_name:
            default_sketch = './dataset/boots/sketch'
            default_original = './dataset/boots/original'
        else:
            print("Could not determine dataset type. Please specify sketch_folder and original_folder.")
            return
        
        if sketch_folder is None:
            sketch_folder = default_sketch
        if original_folder is None:
            original_folder = default_original
    
    print(f"\nUsing dataset folders:")
    print(f"  Sketches: {sketch_folder}")
    print(f"  Originals: {original_folder}")
    
    for i, (exp_name, model_file, output_dir, model_name) in enumerate(generator_models, 1):
        print(f"\n[{i}/{len(generator_models)}] Processing model: {os.path.basename(model_file)}")
        print(f"Output directory: {output_dir}")
        
        try:
            generate_images_from_folder(model_file, sketch_folder, original_folder, output_dir, model_name)
        except Exception as e:
            print(f"Error processing model {model_file}: {str(e)}")
            continue
    
    print(f"\nExperiment {experiment_name} processed!")
    
    # Create models comparison summary
    try:
        print("\nCreating models comparison summary...")
        create_models_comparison_summary(experiment_dir)
    except Exception as e:
        print(f"Error creating models comparison summary: {str(e)}")

# ============================
# Batch prediction function
# ============================
def generate_images_from_folder(generator_model_path, sketch_folder, original_folder, predicted_dir, model_name=None):
    """
    Generate images from all sketches in a folder using the trained model.
    
    Args:
        generator_model_path: Path to the trained generator model
        sketch_folder: Folder containing sketch images
        original_folder: Folder containing original images
        predicted_dir: Output directory for generated images
        model_name: Name of the model (used for file prefixes)
    """
    model = load_model(generator_model_path, compile=False)
    
    # If no model name provided, extract from path
    if model_name is None:
        model_name = os.path.splitext(os.path.basename(generator_model_path))[0]
    
    # Create subdirectories within predictions folder
    os.makedirs(predicted_dir, exist_ok=True)
    
    sketch_files = [f for f in os.listdir(sketch_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not sketch_files:
        print(f"No image files found in {sketch_folder}")
        return
    
    print(f"Found {len(sketch_files)} sketch files. Generating predictions with model {model_name}...")
    
    for i, filename in enumerate(sketch_files, 1):
        sketch_path = os.path.join(sketch_folder, filename)
        original_path = os.path.join(original_folder, filename)
        
        try:
            sketch_array, normalized_sketch = preprocess_sketch(sketch_path)
            
            original = load_img(original_path, target_size=(IMG_SIZE, IMG_SIZE))
            original_array = img_to_array(original)
            
            generated = generate_single_image(model, normalized_sketch)
            
            # Use model name in file naming to distinguish between different models
            base_name = os.path.splitext(filename)[0]
            output_original_path = os.path.join(predicted_dir, f"{base_name}_{model_name}_O.jpg")
            output_sketch_path = os.path.join(predicted_dir, f"{base_name}_{model_name}_S.jpg")
            output_predicted_path = os.path.join(predicted_dir, f"{base_name}_{model_name}_P.jpg")
            
            plt.imsave(output_original_path, original_array.astype(np.uint8))
            plt.imsave(output_sketch_path, sketch_array.astype(np.uint8))
            plt.imsave(output_predicted_path, generated)
            
            print(f"[{i}/{len(sketch_files)}] Generated: {output_predicted_path}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    print(f"Generation complete! Results saved to: {predicted_dir}")

# ============================
# Run prediction
# ============================
if __name__ == '__main__':
    print(f"Running predictions for experiment: {MODEL_NAME}")
    generate_predictions_for_experiment(EXPERIMENT_DIR)
