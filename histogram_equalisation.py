import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

def histogram_equalization(image):
    hist, _ = np.histogram(image.flatten(), 256, [0, 256])
    pmf = hist / np.sum(hist)
    cdf = np.cumsum(pmf)
    equalized_image = (cdf[image] * 255).astype(np.uint8)

    return equalized_image

def display_images_and_histograms(original_image, processed_image, title):
    # Plot histograms
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Histogram - Original Image')
    plt.hist(original_image.flatten(), 256, [0, 256], color='black')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    plt.subplot(2, 3, 4)
    plt.title(title)
    plt.imshow(processed_image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Histogram - ' + title)
    plt.hist(processed_image.flatten(), 256, [0, 256], color='black')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def histogram_stretching(image):
    # Calculate min and max intensity values
    min_intensity = np.min(image)
    max_intensity = np.max(image)

    # Apply histogram stretching
    stretched_image = ((image - min_intensity) / (max_intensity - min_intensity)) * 255

    return stretched_image.astype(np.uint8)

def main():
    (train_images, _), (_, _) = fashion_mnist.load_data()
    random_index = np.random.randint(0, len(train_images))
    #original_image = train_images[random_index]
    original_image = cv2.imread('images/lowcontrast_test2.jpg')
    
    equalized_image = histogram_equalization(original_image)
    display_images_and_histograms(original_image, equalized_image, 'Equalized Image')

    stretched_image = histogram_stretching(original_image)
    display_images_and_histograms(original_image, stretched_image, 'Stretched Image')

if __name__ == "__main__":
    main()
