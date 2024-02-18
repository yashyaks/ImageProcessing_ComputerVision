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
    
def thresholding(image, threshold):
    thresholded_image = (image > threshold) * 255
    return thresholded_image

def main():
    # Choose a random image from the dataset
    random_index = np.random.randint(0, len(train_images))
    original_image = train_images[random_index]

    # Image Thresholding at various levels
    thresholds = [50, 100, 150]
    for threshold in thresholds:
        thresholded_image = thresholding(original_image, threshold)
        display_images(original_image, thresholded_image, f'Original Image (Threshold={threshold})', f'Thresholded Image (Threshold={threshold})')

if __name__ == "__main__":
    main()
