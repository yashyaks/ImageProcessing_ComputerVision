# imread(folder/path_to_image, flag) 
    # flag:
        # cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode

# COLOR_BGR2GRAY vs COLOR_BGR2GRAY
# https://stackoverflow.com/questions/62855718/why-would-cv2-color-rgb2gray-and-cv2-color-bgr2gray-give-different-results


import cv2
img = cv2.imread('images/test.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('grayscale image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

from keras.datasets import mnist
import random
(train_images, train_labels),(train_images, train_labels) = mnist.load_data()
# trying to read images using cv2.imread from the MNIST dataset doesnt work. The MNIST dataset consists of NumPy arrays representing images, and cv2.imread is not suitable for reading images from this dataset.

# also mnist is black and white

# instead use cvtColor to convert the image to grayscale
# cv2.cvtColor() method is used to convert an image from one color space to another.
# Syntax: cv2.cvtColor(src, code[, dst[, dstCn]])

# Parameters:
# src: It is the image whose color space is to be changed.
# code: It is the color space conversion code.
# dst: It is the output image of the same size and depth as src image. It is an optional parameter.
# dstCn: It is the number of channels in the destination image. If the parameter is 0 then the number of the channels is derived automatically from src and code. It is an optional parameter.

# Return Value: It returns an image.

def grayscale_random_images(images):
    #grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.cvtColor(images, cv2.COLOR_RGB2GRAY)
    cv2.imshow("Random Image (Color)", images)
    cv2.imshow("Random Image (Grayscale)", grayscale_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

img2 = cv2.imread('images/test.jpg')
grayscale_random_images(img2)