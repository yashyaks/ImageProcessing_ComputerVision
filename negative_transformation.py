import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import cv2

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
def display_images_cv2(original_image, transformed_image, title1, title2):
    # Display the original and transformed images side by side
    cv2.imshow(title1, original_image)
    cv2.imshow(title2, transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def negative_transformation(image):
    # Perform image negative transformation
    negative_image = 256 - 1 - image
    return negative_image

def main():
    # Choose a random image from the dataset
    random_index = np.random.randint(0, len(train_images))
    original_image = train_images[random_index]
    # Perform image negative transformation
    negative_image = negative_transformation(original_image)
    # Display the original and negative images
    display_images(original_image, negative_image, 'Original Image', 'Negative Image')

if __name__ == "__main__":
    main()
