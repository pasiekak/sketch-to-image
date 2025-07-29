from numpy.random import randint
from keras import initializers, layers, models, optimizers
from keras import Input
from matplotlib import pyplot as plt
import numpy as np
import os
import time

from config import WEIGHT_INIT_STDDEV, LEAKY_RELU_ALPHA, ADAM_LEARNING_RATE, ADAM_BETA_1, ADAM_BETA_2, LOSS_WEIGHT, DROPOUT_RATE, OUTPUT_DIR, EXPERIMENT_NAME, GAN_LOSS_WEIGHTS


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
    x = layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)

    # Final layer: map to 1-dimensional output
    x = layers.Conv2D(1, (4,4), padding='same', kernel_initializer=init)(x)

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
    
    b = layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
    b = layers.Activation('relu')(b)

    # Bottleneck (no need for additional conv, e8 is already 1x1)
    # U-Net decoder with skip connections (channels double due to concatenation)
    d1 = create_decoder_block(b, e7, 512)                      # 1→2 + skip e7 → 512+512=1024
    d2 = create_decoder_block(d1, e6, 512)                      # 2→4 + skip e6 → 512+512=1024  
    d3 = create_decoder_block(d2, e5, 512)                      # 4→8 + skip e5 → 512+512=1024
    d4 = create_decoder_block(d3, e4, 512, dropout=False)       # 8→16 + skip e4 → 512+512=1024
    d5 = create_decoder_block(d4, e3, 256, dropout=False)       # 16→32 + skip e3 → 256+256=512
    d6 = create_decoder_block(d5, e2, 128, dropout=False)       # 32→64 + skip e2 → 128+128=256
    d7 = create_decoder_block(d6, e1, 64, dropout=False)        # 64→128 + skip e1 → 64+64=128

    # Final convolution to get RGB output (128->256)
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
def train(discriminator_model, generator_model, gan_model, dataset, n_epochs=100, n_batch=1, output_dir=None, experiment_name=None):
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
    
    # Use provided output_dir or default from config
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Use provided experiment_name or default from config
    if experiment_name is None:
        experiment_name = EXPERIMENT_NAME
    
    # Combine output_dir and experiment_name
    full_output_dir = os.path.join(output_dir, experiment_name)
    
    n_patch = discriminator_model.output_shape[1]
    sketches, originals = dataset
    
    batch_per_epoch = int(len(sketches) / n_batch)
    
    # Initialize timing variables
    training_start_time = time.time()
    epoch_times = []
    batch_times = []
    
    print(f"Starting training with {n_epochs} epochs, {batch_per_epoch} batches per epoch")
    print("-" * 80)
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        print(f' ========== Epoch {epoch+1}/{n_epochs} ========== ')
        
        epoch_d_loss = []
        epoch_g_loss = []
        
        # Track batch timing for progress reports
        batch_timing_samples = []
        
        # Train on batches
        for batch_idx in range(batch_per_epoch):
            batch_start_time = time.time()
            
            # Generate real samples using existing function
            [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
            
            # Generate fake samples using existing function
            X_fakeB, y_fake = generate_fake_samples(generator_model, X_realA, n_patch)
            X_fake = [X_realA, X_fakeB]
            
            # Train discriminator on real samples
            d_loss_real = discriminator_model.train_on_batch([X_realA, X_realB], y_real)
            
            # Train discriminator on fake samples
            d_loss_fake = discriminator_model.train_on_batch(X_fake, y_fake)
            
            # Train generator (via GAN model)
            # Generator wants discriminator to classify generated images as real
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
            
            # Calculate batch time
            batch_time = time.time() - batch_start_time
            batch_timing_samples.append(batch_time)  # Store all batch times
            
            epoch_d_loss.append([d_loss_real, d_loss_fake])
            epoch_g_loss.append(g_loss)
            
            # Print batch progress
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                # Calculate total time for recent batches
                recent_batches = min(50, len(batch_timing_samples))
                total_recent_time = np.sum(batch_timing_samples[-recent_batches:])
                print(f'Batch {batch_idx+1}/{batch_per_epoch} - D Real: {d_loss_real:.4f} | D Fake: {d_loss_fake:.4f} | G Loss: {g_loss:.4f} | Last {recent_batches} batches: {total_recent_time:.1f}s')
        
        # Calculate epoch timing
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Calculate statistics
        avg_d_loss_real = np.mean([loss[0] for loss in epoch_d_loss])
        avg_d_loss_fake = np.mean([loss[1] for loss in epoch_d_loss])
        avg_g_loss = np.mean(epoch_g_loss)
        
        # Calculate timing statistics
        avg_batch_time = np.mean(batch_timing_samples) if batch_timing_samples else 0
        avg_epoch_time = np.mean(epoch_times)
        total_training_time = time.time() - training_start_time
        eta_minutes = (avg_epoch_time * (n_epochs - epoch - 1)) / 60
        
        # Print epoch summary with timing
        print(f'Epoch {epoch+1} Summary:')
        print(f'  Losses - D Real: {avg_d_loss_real:.4f} | D Fake: {avg_d_loss_fake:.4f} | G Loss: {avg_g_loss:.4f}')
        print(f'  Timing - Epoch: {epoch_time:.1f}s | Avg batch: {avg_batch_time:.2f}s | ETA: {eta_minutes:.1f}min')
        
        # Save model and generate samples every epoch
        summarize_performance(epoch, generator_model, discriminator_model, 
                            dataset, target_dir=f'{full_output_dir}/epoch_{epoch+1}/')
            
        print('-' * 60)
    
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
            

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
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
        # Ensure target_dir ends with slash for proper path joining
        if not target_dir.endswith('/') and not target_dir.endswith('\\'):
            target_dir += '/'
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)  # Create directories recursively
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    # plot real source images
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(X_fakeB[i])
    # plot real target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples*2 + i)
        plt.axis('off')
        plt.imshow(X_realB[i])
    # save plot to file
    filename1 = 'plot_%06d.png' % (step+1)
    plt.savefig(target_dir + filename1)
    plt.close()
    # save the generator model
    g_model.save(target_dir + 'g_model.keras')
    
    # save the discriminator model
    d_model.save(target_dir + 'd_model.keras')
    
    print('>Saved: %s and %s' % (filename1, 'g_model & d_model'))