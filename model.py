from config import ADAM_BETA_1, ADAM_BETA_2, ADAM_LEARNING_RATE, DROPOUT_RATE, EARLY_EXIT_GLOSS_THRESHOLD, EXPERIMENT_NAME, GAN_LOSS_WEIGHTS, LEAKY_RELU_ALPHA, LOSS_WEIGHT, MODEL_SAVE_INTERVAL, OUTPUT_DIR, WEIGHT_INIT_STDDEV
from keras import initializers, layers, models, optimizers
from keras import Input
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import glob
import csv

def create_discriminatior70x70(input_shape):
    """
    Creates a 70x70 PatchGAN discriminator for Pix2Pix.

    Architecture: C64-C128-C256-C512
    - Takes concatenated sketch + target image as input
    - Outputs 70x70 patches classification (real/fake)
    - Uses LeakyReLU activation and BatchNorm (except first layer)

    Args:
        image_shape (tuple): Shape of input images (height, width, channels)

    Returns:
        keras.Model: Compiled discriminator model
    """
    # Weight initializer: normal distribution (mean=0.0, std=0.02) for stable GAN training
    init = initializers.RandomNormal(stddev=WEIGHT_INIT_STDDEV)

    input_source_image = Input(shape=input_shape) # Original sketch
    input_target_image = Input(shape=input_shape) # Generated image

    merged = layers.Concatenate()([input_source_image, input_target_image])

    # C64-C128-C256-C512

    # C64: 4x4 size with stride (2,2) = 2 steps right, 2 steps down
    x = layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    x = layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)

    # C128: 4x4 size with stride (2,2) = 2 steps right, 2 steps down
    x = layers.Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)

    # C256: 4x4 size with stride (2,2) = 2 steps right, 2 steps down
    x = layers.Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)

    # C512: 4x4 size with stride (2,2) = 2 steps right, 2 steps down
    x = layers.Conv2D(512, (4,4), strides=(1,1), padding='same', kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)

    # Final layer: map to 1-dimensional output
    x = layers.Conv2D(1, (4,4), strides=(1,1), padding='same', kernel_initializer=init)(x)

    # Sigmoid
    output = layers.Activation('sigmoid')(x)

    # Create model
    model = models.Model([input_source_image, input_target_image], output, name='Discriminator')

    # Model compiling
    optimizer = optimizers.Adam(learning_rate=ADAM_LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, loss_weights=[LOSS_WEIGHT])

    return model

def create_generator(image_shape):
    """
    Creates a U-Net generator for Pix2Pix GAN.

    Architecture: Encoder-Decoder with skip connections
    - Encoder: C64-C128-C256-C512-C512-C512-C512-C512 (downsample sketch)
    - Decoder: CD512-CD512-CD512-C512-C256-C128-C64 (upsample to realistic image)
    - Skip connections preserve fine details from input sketch
    - Uses ReLU in decoder, LeakyReLU in encoder
    - Dropout applied to first 3 decoder layers for regularization

    Args:
        image_shape (tuple): Shape of input images (height, width, channels)

    Returns:
        keras.Model: Compiled generator model that transforms sketches to images
    """

    # Weight initializer: normal distribution (mean=0.0, std=0.02) for stable GAN training
    init = initializers.RandomNormal(stddev=WEIGHT_INIT_STDDEV)

    # Image input
    input_image = Input(shape=image_shape)

    # Encoder model C64-C128-C256-C512-C512-C512-C512-C512
    e1 = create_encoder_block(input_image, 64, batchNorm=False)  # 256→128
    e2 = create_encoder_block(e1, 128)                           # 128→64
    e3 = create_encoder_block(e2, 256)                           # 64→32
    e4 = create_encoder_block(e3, 512)                           # 32→16
    e5 = create_encoder_block(e4, 512)                           # 16→8
    e6 = create_encoder_block(e5, 512)                           # 8→4
    e7 = create_encoder_block(e6, 512)                           # 4→2
    e8 = create_encoder_block(e7, 512)                           # 2→1

    # Bottleneck (no need for additional conv, e8 is already 1x1)
    b = layers.Activation('relu')(e8)

    # U-Net decoder with skip connections (channels double due to concatenation)
    d1 = create_decoder_block(b, e7, 512)                        # 1→2 + skip e7 → 512+512=1024
    d2 = create_decoder_block(d1, e6, 512)                       # 2→4 + skip e6 → 512+512=1024
    d3 = create_decoder_block(d2, e5, 512)                       # 4→8 + skip e5 → 512+512=1024
    d4 = create_decoder_block(d3, e4, 512, dropout=False)        # 8→16 + skip e4 → 512+512=1024
    d5 = create_decoder_block(d4, e3, 256, dropout=False)        # 16→32 + skip e3 → 256+256=512
    d6 = create_decoder_block(d5, e2, 128, dropout=False)        # 32→64 + skip e2 → 128+128=256
    d7 = create_decoder_block(d6, e1, 64, dropout=False)         # 64→128 + skip e1 → 64+64=128

    # Final convolution to get RGB output (128→256)
    g = layers.Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    output_image = layers.Activation('tanh')(g)

    model = models.Model(input_image, output_image, name='Generator')
    return model

# Create encoder block to be used in generator
def create_encoder_block(layer_in, n_filters, batchNorm=True):
    # Weight initializer: normal distribution (mean=0.0, std=0.02) for stable GAN training
    init = initializers.RandomNormal(stddev=WEIGHT_INIT_STDDEV)

    x = layers.Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)

    # Batch normalization if needed
    if batchNorm:
        x = layers.BatchNormalization()(x, training=True)

    # Leaky ReLU activation
    x = layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
    return x

# Create a decoder block to be used in generator
def create_decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # Weight initializer: normal distribution (mean=0.0, std=0.02) for stable GAN training
    init = initializers.RandomNormal(stddev=WEIGHT_INIT_STDDEV)

    x = layers.Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)

    # Batch normalization
    x = layers.BatchNormalization()(x, training=True)

    # Disable half of neurons during training
    if (dropout):
        x = layers.Dropout(DROPOUT_RATE)(x, training=True)

    # Concatenate
    x = layers.Concatenate()([x, skip_in])

    # Activation
    x = layers.Activation('relu')(x)

    return x

# Create GAN
def create_gan(generator_model, discriminator_model, image_shape):
    for layer in discriminator_model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    input_source = Input(shape=image_shape)
    generator_output = generator_model(input_source)
    discriminator_output = discriminator_model([input_source, generator_output])

    model = models.Model(input_source, [discriminator_output, generator_output], name='GAN')

    opt = optimizers.Adam(learning_rate=ADAM_LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2)

    model.compile(loss=['binary_crossentropy', 'mae'],
                  optimizer=opt, loss_weights=GAN_LOSS_WEIGHTS)

    return model

# Train pix2pix models
def train(discriminator_model, generator_model, gan_model, dataset, n_epochs=100, n_batch=1, output_dir=OUTPUT_DIR, experiment_name=EXPERIMENT_NAME):
    """
    Train pix2pix GAN models.

    Args:
        discriminator_model: Compiled discriminator model
        generator_model: Compiled generator model
        gan_model: Combined GAN model
        dataset: Tuple of (sketches, originals) arrays
        n_epochs: Number of training epochs
        n_batch: Batch size for training
        output_dir: Directory to save models and images (defaults to config.OUTPUT_DIR)
        experiment_name: Name of experiment subdirectory (defaults to config.EXPERIMENT_NAME)
    """

    # Combine output_dir and experiment_name
    full_output_dir = os.path.join(output_dir, experiment_name)

    n_patch = discriminator_model.output_shape[1]
    # Use only training data for training loop
    train_sketches, train_originals = dataset[0], dataset[1]
    batch_per_epoch = int(len(train_sketches) / n_batch)

    # Select a single sketch-original pair at the start of training
    idx = np.random.randint(0, train_sketches.shape[0])
    fixed_sketch = train_sketches[idx:idx+1]
    fixed_original = train_originals[idx:idx+1]

    # Create directory for generated images only
    generated_dir = os.path.join(full_output_dir, "generated")
    os.makedirs(generated_dir, exist_ok=True)

    # Save sketch and original only once directly in experiment folder
    sketch_path = os.path.join(full_output_dir, "sketch.png")
    original_path = os.path.join(full_output_dir, "original.png")
    if not os.path.exists(sketch_path):
        plt.imsave(sketch_path, (fixed_sketch[0] + 1) / 2.0)
    if not os.path.exists(original_path):
        plt.imsave(original_path, (fixed_original[0] + 1) / 2.0)

    # Initialize timing variables
    training_start_time = time.time()
    epoch_times = []
    batch_times = []
    statistics_data = []  # List to store statistics for each epoch

    print(f"Starting training with {n_epochs} epochs, {batch_per_epoch} batches per epoch")
    print("-" * 80)

    # Create models subdirectory
    models_dir = os.path.join(full_output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Save initial models (epoch 0) before training starts
    g_model_path_initial = os.path.join(models_dir, 'g_model_epoch_000.keras')
    d_model_path_initial = os.path.join(models_dir, 'd_model_epoch_000.keras')
    generator_model.save(g_model_path_initial, overwrite=True)
    discriminator_model.save(d_model_path_initial, overwrite=True)
    print(f"Initial models saved (epoch 0)")

    for epoch in range(n_epochs):
        epoch_start_time = time.time()

        epoch_d_loss = []
        epoch_g_loss = []
        batch_timing_samples = []

        for batch_idx in range(batch_per_epoch):
            batch_start_time = time.time()
            # Only use training data for real samples
            [X_realA, X_realB], y_real = generate_real_samples((train_sketches, train_originals), n_batch, n_patch)
            X_fakeB, y_fake = generate_fake_samples(generator_model, X_realA, n_patch)
            X_fake = [X_realA, X_fakeB]
            d_loss_real = discriminator_model.train_on_batch([X_realA, X_realB], y_real)
            d_loss_fake = discriminator_model.train_on_batch(X_fake, y_fake)
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
            batch_time = time.time() - batch_start_time
            batch_timing_samples.append(batch_time)
            epoch_d_loss.append([d_loss_real, d_loss_fake])
            epoch_g_loss.append(g_loss)

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_d_loss_real = np.mean([loss[0] for loss in epoch_d_loss])
        avg_d_loss_fake = np.mean([loss[1] for loss in epoch_d_loss])
        avg_g_loss = np.mean(epoch_g_loss)
        avg_batch_time = np.mean(batch_timing_samples) if batch_timing_samples else 0
        avg_epoch_time = np.mean(epoch_times)
        total_training_time = time.time() - training_start_time
        eta_minutes = (avg_epoch_time * (n_epochs - epoch - 1)) / 60
        print(f'Epoch {epoch+1}/{n_epochs}: D Real: {avg_d_loss_real:.4f} | D Fake: {avg_d_loss_fake:.4f} | G Loss: {avg_g_loss:.4f} | Time: {epoch_time:.1f}s | ETA: {eta_minutes:.1f}min | Batches: {batch_per_epoch} | Total images in training: {len(train_sketches)}')

        # Collect statistics for this epoch
        epoch_stats = {
            'epoch': epoch + 1,
            'd_loss_real': avg_d_loss_real,
            'd_loss_fake': avg_d_loss_fake,
            'g_loss': avg_g_loss,
            'epoch_time': epoch_time,
            'total_time': total_training_time,
            'eta_minutes': eta_minutes
        }
        statistics_data.append(epoch_stats)

        # Save generated image for the same pair in generated_dir
        summarize_performance(epoch, generator_model, discriminator_model,
                      (fixed_sketch, fixed_original), target_dir=generated_dir)

        # Save models every N epochs (based on MODEL_SAVE_INTERVAL) and on the last epoch OR on early exit
        should_save_regular = (MODEL_SAVE_INTERVAL > 0 and (epoch + 1) % MODEL_SAVE_INTERVAL == 0) or epoch == n_epochs - 1
        should_save_early_exit = avg_g_loss < EARLY_EXIT_GLOSS_THRESHOLD
        
        if should_save_regular or should_save_early_exit:
            if should_save_early_exit:
                # Early exit - add suffix to filename
                g_model_path = os.path.join(models_dir, f'g_model_epoch_{epoch+1:03d}_early_exit.keras')
                d_model_path = os.path.join(models_dir, f'd_model_epoch_{epoch+1:03d}_early_exit.keras')
                print(f"Early exit triggered at epoch {epoch+1} (G Loss: {avg_g_loss:.4f} < {EARLY_EXIT_GLOSS_THRESHOLD})")
            else:
                # Regular save
                g_model_path = os.path.join(models_dir, f'g_model_epoch_{epoch+1:03d}.keras')
                d_model_path = os.path.join(models_dir, f'd_model_epoch_{epoch+1:03d}.keras')
            
            generator_model.save(g_model_path, overwrite=True)
            discriminator_model.save(d_model_path, overwrite=True)
            print(f"Models saved at epoch {epoch+1}")
        
        # Exit if early exit threshold reached
        if should_save_early_exit:
            break

    # Final training summary
    total_training_time = time.time() - training_start_time
    total_hours = total_training_time / 3600
    avg_epoch_time_final = np.mean(epoch_times)
    print(f"\nTraining completed!")
    print(f"Total training time: {total_hours:.2f} hours ({total_training_time:.1f} seconds)")
    print(f"Average time per epoch: {avg_epoch_time_final:.1f} seconds")
    print(f"Total epochs completed: {n_epochs}")
    print(f"Total batches processed: {n_epochs * batch_per_epoch}")
    print("=" * 80)

    # After training: generate summary plot with sketch, original, last generated
    summary_plot_path = os.path.join(full_output_dir, "summary_last_epoch.png")
    sketch_img = (fixed_sketch[0] + 1) / 2.0
    original_img = (fixed_original[0] + 1) / 2.0
    generated_img = generator_model.predict(fixed_sketch, verbose=0)
    generated_img = (generated_img[0] + 1) / 2.0

    save_summary_plot(sketch_img, original_img, generated_img, f"epoch {n_epochs}", summary_plot_path)

    # Generate and save training statistics
    generate_statistics(output_dir, experiment_name, statistics_data)



# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = np.random.randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples, verbose=0)
    # create 'fake' class labels (0)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

# generate samples and save as a plot and save the model
# GAN models do not converge, we just want to find a good balance between
# the generator and the discriminator. Therefore, it makes sense to periodically
# save the generator model and check how good the generated image looks.
def summarize_performance(step, g_model, d_model, dataset, target_dir='', n_samples=3):
    if target_dir:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

    # dataset: (fixed_sketch, fixed_original), batch=1
    X_realA, X_realB = dataset
    # generate image
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    X_fakeB = (X_fakeB[0] + 1) / 2.0

    # Save generated image for this epoch
    filename_gen = f"generated_epoch_{step+1:03d}.png"
    plt.imsave(os.path.join(target_dir, filename_gen), X_fakeB)

def save_summary_plot(sketch_img, original_img, generated_img, title_suffix, output_path):
    """
    Save a side-by-side summary plot showing sketch, original and generated images.

    Args:
        sketch_img (np.array): Input sketch image, scaled to [0,1]
        original_img (np.array): Ground truth image, scaled to [0,1]
        generated_img (np.array): Image generated by the model, scaled to [0,1]
        title_suffix (str): Text to append in title of generated image (e.g., 'epoch 5')
        output_path (str): Path to save the resulting .png image
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title('Sketch')
    plt.imshow(sketch_img)

    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title('Original')
    plt.imshow(original_img)

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title(f'Generated ({title_suffix})')
    plt.imshow(generated_img)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Summary saved: {output_path}")

def generate_statistics(output_dir, experiment_name, epoch_data):
    """
    Generate and save training statistics to CSV file.
    
    Args:
        output_dir (str): Base output directory
        experiment_name (str): Name of the experiment
        epoch_data (list): List of dictionaries containing epoch statistics
                          Each dict should have keys: epoch, d_loss_real, d_loss_fake, 
                          g_loss, epoch_time, total_time
    """
    full_output_dir = os.path.join(output_dir, experiment_name)
    stats_file = os.path.join(full_output_dir, "training_statistics.csv")
    
    # Ensure directory exists
    os.makedirs(full_output_dir, exist_ok=True)
    
    # CSV headers
    headers = [
        'epoch', 
        'discriminator_loss_real', 
        'discriminator_loss_fake', 
        'generator_loss',
        'epoch_time_seconds',
        'total_training_time_seconds',
        'eta_minutes'
    ]
    
    # Write or append to CSV file
    file_exists = os.path.exists(stats_file)
    
    with open(stats_file, 'a' if file_exists else 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        
        # Write header only if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        # Write data
        for data in epoch_data:
            writer.writerow({
                'epoch': data.get('epoch', 0),
                'discriminator_loss_real': data.get('d_loss_real', 0.0),
                'discriminator_loss_fake': data.get('d_loss_fake', 0.0),
                'generator_loss': data.get('g_loss', 0.0),
                'epoch_time_seconds': data.get('epoch_time', 0.0),
                'total_training_time_seconds': data.get('total_time', 0.0),
                'eta_minutes': data.get('eta_minutes', 0.0)
            })
    
    print(f"Training statistics saved to: {stats_file}")