import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

(train_images, _), (_, _) = fashion_mnist.load_data()

def display_images(original_image, transformed_image, title1, title2):
    # Display the original and transformed images side by side
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title(title1)

    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image, cmap='gray')
    plt.title(title2)

    plt.show()

def gray_level_slicing(image, lower_bound, upper_bound, background_intensity=False):
    # Perform gray-level slicing
    if background_intensity:
        sliced_image = np.where((image >= lower_bound) & (image <= upper_bound), 255, image)
    else:
        sliced_image = np.where((image >= lower_bound) & (image <= upper_bound), image, 255)
    return sliced_image

def main():
    # Choose a random image from the dataset
    random_index = np.random.randint(0, len(train_images))
    original_image = train_images[random_index]

    # Gray-level slicing with and without background intensity slicing
    lower_bound = 100
    upper_bound = 200
    
    # With background intensity slicing
    sliced_image_with_background = gray_level_slicing(original_image, lower_bound, upper_bound, True)
    display_images(original_image, sliced_image_with_background, 'Original Image', 'Gray-level Sliced Image (With Background Slicing)')

    # Without background intensity slicing
    sliced_image_without_background = gray_level_slicing(original_image, lower_bound, upper_bound, False)
    display_images(original_image, sliced_image_without_background, 'Original Image', 'Gray-level Sliced Image (Without Background Slicing)')

if __name__ == "__main__":
    main()
