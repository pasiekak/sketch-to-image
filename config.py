# Set to False to force CPU only, True to use GPU if available
USE_GPU = True
IMG_SIZE = 256 

TRAINING_DATASET_PATH = "./dataset/faces_all/train"

# Dataset configuration
USE_AUGMENTATION = False            # Set to False to use only original sketches, True to include augmented data
RANDOM_SEED = 42                   # For reproducible random sampling
MAX_TRAINING_SAMPLES = 20000

# Output directory for models and generated images
OUTPUT_DIR = './models'

# PIX2PIX - Parameters
WEIGHT_INIT_STDDEV = 0.02     # Normal weight initialization std dev (base: 0.02)
LEAKY_RELU_ALPHA = 0.2        # LeakyReLU negative slope (base: 0.2)
ADAM_LEARNING_RATE = 0.0001   # Adam optimizer learning rate (base: 0.0002)
ADAM_BETA_1 = 0.5             # Adam momentum for gradient (base: 0.5, default: 0.9)
ADAM_BETA_2 = 0.999           # Adam momentum for squared gradient (base: 0.999)
LOSS_WEIGHT = 0.5             # Binary crossentropy loss weight (base: 0.5)
DROPOUT_RATE = 0.25            # Dropout rate for generator decoder (base: 0.5)
BATCH_SIZE = 1                # Number of samples processed per training step (base: 1)
EPOCHS = 100
MODEL_SAVE_INTERVAL = 1      # Save model every N epochs (0 = save only at the end)
GAN_LOSS_WEIGHTS = [1, 200]  # [binary_crossentropy, mae] weights for generator training
EARLY_EXIT_GLOSS_THRESHOLD = 0 # 0 - never early leave

EXPERIMENT_NAME = (
    f"#FACES_ALL_S{MAX_TRAINING_SAMPLES}_E{EPOCHS}_B{BATCH_SIZE}_"
    f"LR{ADAM_LEARNING_RATE}_B1{ADAM_BETA_1}_B2{ADAM_BETA_2}_"
    f"LRA{LEAKY_RELU_ALPHA}_DO{DROPOUT_RATE}_GW{GAN_LOSS_WEIGHTS[0]}_L1W{GAN_LOSS_WEIGHTS[1]}_"
    f"MSI{MODEL_SAVE_INTERVAL}_AUG{int(USE_AUGMENTATION)}"
)