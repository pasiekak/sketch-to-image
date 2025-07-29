from matplotlib import pyplot as plt
from scripts.load_dataset import load_dataset
import scripts.utils as utils
import scripts.pix_2_pix as p2p
import config

if __name__ == '__main__':
    # Load dataset with sample limiting
    print(f"Loading dataset - category: {config.TRAINING_CATEGORY}, max samples: {config.MAX_TRAINING_SAMPLES}")
    dataset = load_dataset(category=config.TRAINING_CATEGORY, max_samples=config.MAX_TRAINING_SAMPLES)
    print(f"Loaded {len(dataset)} image pairs")
    
    image_shape = dataset[0].shape[1:]
    dataset = utils.preprocess_data(dataset)

    # Create models
    print("Creating discriminator...")
    discriminator_model = p2p.create_discriminatior70x70(image_shape)
    
    print("Creating generator...")
    generator_model = p2p.create_generator(image_shape)

    print("Creating GAN...")
    gan_model = p2p.create_gan(generator_model, discriminator_model, image_shape)

    print("GAN SUMMARY")
    gan_model.summary()

    print("Starting training...")
    p2p.train(discriminator_model, generator_model, gan_model, dataset, n_epochs=config.EPOCHS, n_batch=config.BATCH_SIZE)
