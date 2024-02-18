import cv2
import numpy as np
import matplotlib.pyplot as plt

def power_law_transformation(image, gamma):
    power_law_image = np.power(image / 255.0, gamma) * 255.0

    return power_law_image.astype(np.uint8)

def display_image_with_histogram(original_image, transformed_image):
    # Plot histograms
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('Histogram - Original Image')
    plt.hist(original_image.flatten(), 256, [0, 256], color='black')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 3)
    plt.title('Power-law Transformed Image')
    plt.imshow(transformed_image)
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('Histogram - Power-law Transformed Image')
    plt.hist(transformed_image.flatten(), 256, [0, 256], color='black')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def main():
    original_image = cv2.imread('images/lowcontrast_test.jpg')

    # Define the gamma value for power-law transformation
    gamma = 0.5
    power_law_transformed_image = power_law_transformation(original_image, gamma)

    # Display images and histograms using the function
    display_image_with_histogram(original_image, power_law_transformed_image)

if __name__ == "__main__":
    main()
