import cv2
import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching(image):
    # Apply contrast stretching
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    stretched_image = ((image - min_intensity) / (max_intensity - min_intensity)) * 255

    return stretched_image.astype(np.uint8)

def display_image_with_histogram(original_image, stretched_image):
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
    plt.title('Stretched Image')
    plt.imshow(stretched_image)
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('Histogram - Stretched Image')
    plt.hist(stretched_image.flatten(), 256, [0, 256], color='black')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def main():
    
    # Load the image
    original_image = cv2.imread('images/lowcontrast_test2.jpg')
    
    # Apply contrast stretching
    stretched_image = contrast_stretching(original_image)

    # Display images and histograms using the function
    display_image_with_histogram(original_image, stretched_image)
    
if __name__ == "__main__":
    main()