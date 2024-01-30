# imread() helps us read an image
# imshow() displays an image in a window
# imwrite() writes an image into the file directory

# imread(folder/path_to_image, flag) 
    # flag argument sepcifies the way in which the image should be read.
        # cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
        # cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
        # cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel
        # Instead of these three flags, you can simply pass integers 1, 0 or -1 respectively.

# imshow(window_name, image)
# imwrite(filename, image)

import cv2
img = cv2.imread('images/test.jpg')
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# read write random images from MNIST dataset
from keras.datasets import mnist
import random
(train_images, train_labels),(train_images, train_labels) = mnist.load_data()

def read_random_images(images):
    random_number = random.randint(1, len(images)-1)
    cv2.imshow("Random Image",images[random_number]) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

read_random_images(train_images)