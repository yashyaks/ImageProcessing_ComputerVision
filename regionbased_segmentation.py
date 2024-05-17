import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

def region_growing(image, seed, threshold):
    # Create a binary output image
    output = np.zeros_like(image)
    # Create a queue for pixels to be processed
    queue = []
    queue.append(seed)
    while queue:
        # Pop the pixel from the queue
        x, y = queue.pop(0)
        # Check if pixel is within image boundaries
        if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
            # Check if the pixel intensity difference is less than threshold
            if abs(int(image[x, y]) - np.mean(output)) <= threshold:
                # Set the pixel in the output image
                output[x, y] = 255
                # Add neighboring pixels to the queue
                queue.append((x + 1, y))
                queue.append((x - 1, y))
                queue.append((x, y + 1))
                queue.append((x, y - 1))
    return output

def main():
    # Load Fashion MNIST dataset
    (X_train, _), (_, _) = fashion_mnist.load_data()
    # Select a random image from the dataset
    image = X_train[np.random.randint(0, len(X_train))]
    image = cv2.imread('images/test.jpg', 0)
    # Convert to single channel 8-bit image
    image = cv2.convertScaleAbs(image)

    # Region growing parameters
    seed = (10, 10)  # Seed pixel position
    threshold = 20   # Intensity threshold

    # Perform region growing
    segmented_image = region_growing(image, seed, threshold)

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(segmented_image, cmap='gray')
    axs[1].set_title('Region Growing Segmentation')
    plt.show()

if __name__ == "__main__":
    main()