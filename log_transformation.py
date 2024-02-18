import cv2
import numpy as np
import matplotlib.pyplot as plt

def log_transformation(image):
    # Apply log transformation
    log_image = np.log1p(image)

    # Normalize to 0-255 range
    log_image = (log_image / np.max(log_image)) * 255

    return log_image.astype(np.uint8)

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
    plt.title('Log Transformed Image')
    plt.imshow(transformed_image)
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('Histogram - Log Transformed Image')
    plt.hist(transformed_image.flatten(), 256, [0, 256], color='black')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def main():
    # Load the image
    original_image = cv2.imread('images/lowcontrast_test2.jpg')

    # Apply log transformation
    log_transformed_image = log_transformation(original_image)

    # Display images and histograms using the function
    display_image_with_histogram(original_image, log_transformed_image)

if __name__ == "__main__":
    main()
