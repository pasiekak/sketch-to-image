import os

from matplotlib import pyplot as plt

from constants import EPOCHS, TESTING_GENERATED_IMAGES_PATH
from scripts.load_dataset import load_dataset
from scripts.pix_2_pix import build_generator, build_discriminator, train_step

if __name__ == '__main__':
    train_dataset, test_dataset = load_dataset()

    generator = build_generator()
    generator.summary()

    discriminator = build_discriminator()
    discriminator.summary()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")

        for sketch_batch, original_batch in train_dataset:
            train_step(sketch_batch, original_batch, generator, discriminator)

        if (epoch + 1) % 10 == 0:
            print(f"Saving generated images for epoch {epoch + 1}")

            for i, (sketch_batch, original_batch) in enumerate(test_dataset):
                generated_images = generator(sketch_batch, training=False)

                os.makedirs(os.path.join(TESTING_GENERATED_IMAGES_PATH, "generated"), exist_ok=True)

                generated_image = generated_images[0]

                generated_image = (generated_image * 0.5) + 0.5
                generated_image = generated_image.numpy()

                image_filename = f"generated_epoch_{epoch + 1}_image_{i}_0.png"
                image_path = os.path.join(TESTING_GENERATED_IMAGES_PATH, "generated", image_filename)

                plt.imshow(generated_image)
                plt.axis('off')
                plt.savefig(image_path)
                plt.close()

            print(f"Generated images saved for epoch {epoch + 1}.")

