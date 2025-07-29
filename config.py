IMG_SIZE = 256

TRAINING_ORIGINAL_IMAGES_PATH = './dataset/raw/original'
TRAINING_SKETCH_IMAGES_PATH = './dataset/raw/sketch'

# Dataset configuration
TRAINING_CATEGORY = 'bicycle'  # Category to load for training (e.g., 'airplane', 'cat', 'car_(sedan)', or 'all' for all categories)
MAX_TRAINING_SAMPLES = 400     # Limit number of samples for quick testing (None = all samples)
RANDOM_SEED = 42               # For reproducible random sampling

# Output directory for models and generated images
OUTPUT_DIR = './output'
EXPERIMENT_NAME = 'models_on_raw_dataset_#3'  # Subdirectory name for current experiment

# PIX2PIX - Parameters
WEIGHT_INIT_STDDEV = 0.02     # Normal weight initialization std dev (base: 0.02)
LEAKY_RELU_ALPHA = 0.2        # LeakyReLU negative slope (base: 0.2)
ADAM_LEARNING_RATE = 0.0002   # Adam optimizer learning rate (base: 0.0002)
ADAM_BETA_1 = 0.5             # Adam momentum for gradient (base: 0.5, default: 0.9)
ADAM_BETA_2 = 0.999           # Adam momentum for squared gradient (base: 0.999)
LOSS_WEIGHT = 0.5             # Binary crossentropy loss weight (base: 0.5)
DROPOUT_RATE = 0.5            # Dropout rate for generator decoder (base: 0.5)
BATCH_SIZE = 4                # Number of samples processed per training step (base: 1)
EPOCHS = 50           

# GAN Loss weights [adversarial_loss_weight, L1_loss_weight]
GAN_LOSS_WEIGHTS = [1, 50]  # [binary_crossentropy, mae] weights for generator training