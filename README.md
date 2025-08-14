# Sketch-to-Image Generator (Pix2Pix GAN)

This project implements a Pix2Pix GAN model for converting sketches to realistic images. The model is trained on face datasets and can generate realistic face images from hand-drawn or computer-generated sketches.

## Project Structure

```
├── main.py                      # Main training script
├── config.py                    # Configuration parameters
├── model.py                     # GAN model definitions (Generator, Discriminator)
├── utils.py                     # Dataset loading and preprocessing utilities
├── predict.py                   # Image prediction and batch processing
├── requirements.txt             # Python dependencies
├── image_to_sketch.py           # Basic sketch generation from images
├── image_to_sketch_realistic.py # Advanced sketch generation algorithm
├── extrude_background.py        # Background removal tool using rembg
├── augment.py                   # Data augmentation utilities
├── crop_images.py               # Image cropping utilities
├── expand_images.py             # Image expansion utilities
├── resize_to_256.py             # Image resizing to 256x256
├── dataset/                     # Training datasets
├── models/                      # Saved models and experiment results
├── test/                        # Test images
└── rembg/                       # Background removal working directory
```

## Installation

1. **Clone the repository** (or download the project files)

2. **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Install additional dependencies for background removal:**
    ```bash
    pip install rembg
    ```

## Dataset Preparation

### 1. Dataset Structure

Organize your dataset in the following structure:

```
dataset/
├── faces_men/
│   ├── train/
│   │   ├── original/     # Original face images
│   │   ├── sketch/       # Corresponding sketches
│   │   └── augmented/    # (Optional) Augmented sketches
│   ├── test/
│   │   ├── original/
│   │   └── sketch/
│   └── validation/
│       ├── original/
│       └── sketch/
```

### 2. Generate Sketches from Images

Convert your original images to sketches using provided tools:

**Basic sketch generation:**

```bash
python image_to_sketch.py
```

**Advanced realistic sketch generation:**

```bash
python image_to_sketch_realistic.py
```

Edit the paths in these scripts to match your dataset structure.

### 3. Background Removal (Optional)

Remove backgrounds from images using the rembg tool:

```bash
python extrude_background.py
```

### 4. Image Preprocessing

Resize images to 256x256 pixels:

```bash
python resize_to_256.py
```

## Configuration

Edit `config.py` to customize training parameters:

### Key Parameters

```python
# Hardware configuration
USE_GPU = True                    # Use GPU if available

# Dataset configuration
TRAINING_DATASET_PATH = "./dataset/faces_men/train"
USE_AUGMENTATION = True           # Include augmented data in training
MAX_TRAINING_SAMPLES = 1000       # Limit training samples
RANDOM_SEED = 42                  # For reproducible results

# Training parameters
BATCH_SIZE = 1                    # Batch size (recommended: 1 for Pix2Pix)
EPOCHS = 20                       # Number of training epochs
ADAM_LEARNING_RATE = 0.0002       # Learning rate
MODEL_SAVE_INTERVAL = 1           # Save model every N epochs

# Model architecture
LEAKY_RELU_ALPHA = 0.2            # LeakyReLU negative slope
DROPOUT_RATE = 0.5                # Dropout rate in generator
GAN_LOSS_WEIGHTS = [1, 100]       # [adversarial_loss, L1_loss] weights
```

## Training

### 1. Basic Training

Run the training script:

```bash
python main.py
```

### 2. Training with Different Configurations

Modify `config.py` parameters and run training:

-   **Without augmentation:** Set `USE_AUGMENTATION = False`
-   **More training samples:** Increase `MAX_TRAINING_SAMPLES`
-   **Longer training:** Increase `EPOCHS`
-   **Different dataset:** Change `TRAINING_DATASET_PATH`

### 3. Monitoring Training

The training will:

-   Display progress in the console
-   Save models every N epochs (controlled by `MODEL_SAVE_INTERVAL`)
-   Generate sample images during training
-   Save training statistics and summary plots

## Model Output

Training creates an experiment directory in `models/` with the following structure:

```
models/
├── #EXPERIMENT_NAME/
│   ├── d_model.keras           # Final discriminator model
│   ├── g_model.keras           # Final generator model
│   ├── sketch.png              # Sample input sketch
│   ├── original.png            # Sample target image
│   ├── summary_last_epoch.png  # Final training summary
│   ├── training_statistics.csv # Training metrics
│   ├── generated/              # Sample outputs per epoch
│   │   ├── generated_epoch_001.png
│   │   ├── generated_epoch_002.png
│   │   └── ...
│   └── models/                 # Model checkpoints
│       ├── g_model_epoch_000.keras
│       ├── d_model_epoch_001.keras
│       └── ...
```

## Prediction/Inference

### 1. Single Image Prediction

Use the `predict.py` script for generating images from sketches:

```python
from predict import generate_image_from_path

model_path = './models/EXPERIMENT_NAME/g_model.keras'
sketch_path = './test/test_sketch.jpg'

original, generated = generate_image_from_path(model_path, sketch_path)
```

### 2. Batch Prediction

Process multiple images at once:

```python
from predict import generate_images_from_folder

generator_model_path = './models/EXPERIMENT_NAME/g_model.keras'
sketch_folder = './test/sketches/'
original_folder = './test/originals/'  # Optional, for comparison
output_folder = './test/predictions/'

generate_images_from_folder(generator_model_path, sketch_folder, original_folder, output_folder)
```

### 3. Model Comparison

Generate comparison summaries across different training epochs:

```python
from predict import create_models_comparison_summary

experiment_dir = './models/EXPERIMENT_NAME/'
create_models_comparison_summary(experiment_dir, num_samples=5)
```

## Utility Scripts

### Image Processing

-   **`crop_images.py`** - Crop images to specific regions
-   **`expand_images.py`** - Expand image datasets
-   **`resize_to_256.py`** - Resize images to 256x256 pixels

### Data Augmentation

-   **`augment.py`** - Apply various augmentation techniques to increase dataset size

## Example Workflow

1. **Prepare your dataset:**

    ```bash
    # Resize images
    python resize_to_256.py

    # Generate sketches
    python image_to_sketch_realistic.py
    ```

2. **Configure training:**
   Edit `config.py` with your desired parameters

3. **Train the model:**

    ```bash
    python main.py
    ```

4. **Generate predictions:**
    ```bash
    python predict.py
    ```

## Tips and Best Practices

### Training Tips

-   Start with smaller datasets (1000-2000 samples) for initial experiments
-   Use `BATCH_SIZE = 1` for Pix2Pix models
-   Monitor the generated images in the `generated/` folder during training
-   Training typically takes several hours depending on dataset size and hardware

### Dataset Quality

-   Ensure sketch-original pairs are properly aligned
-   Use consistent image sizes (256x256 recommended)
-   Higher quality original images produce better results
-   Diverse sketching styles improve model generalization

### Hardware Requirements

-   **GPU recommended** for training (CUDA-compatible)
-   **CPU training possible** but significantly slower
-   **RAM:** 8GB minimum, 16GB+ recommended for larger datasets
-   **Storage:** Several GB for models and generated images

## Troubleshooting

### Common Issues

**CUDA/GPU Issues:**

```bash
# Force CPU-only training
# Set USE_GPU = False in config.py
```

**Memory Issues:**

```bash
# Reduce batch size or training samples
# Set BATCH_SIZE = 1 and reduce MAX_TRAINING_SAMPLES in config.py
```

**Dataset Loading Errors:**

-   Check dataset path in `config.py`
-   Ensure proper folder structure
-   Verify image file formats (jpg, png supported)

**Model Loading Errors:**

-   Ensure model files exist in the specified paths
-   Check file permissions
-   Verify Keras/TensorFlow compatibility

## Requirements

-   Python 3.7+
-   TensorFlow 2.x
-   OpenCV
-   Pillow (PIL)
-   NumPy
-   Matplotlib
-   rembg (for background removal)

See `requirements.txt` for complete dependency list.

## License

This project is for educational and research purposes. Please respect the licenses of the datasets and libraries used.
