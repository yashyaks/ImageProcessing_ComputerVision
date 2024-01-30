import cv2

first_image = cv2.imread('images/drake_1.jpeg')
second_image = cv2.imread('images/drake_2.jpeg')
def close():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def adding(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for addition.")
    added_image = cv2.add(image1, image2)
    cv2.imshow("Image1", first_image)
    cv2.imshow("Image2", second_image)
    cv2.imshow("Added Image", added_image)

def subtracting(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for addition.")
    subtracted_image = cv2.subtract(image1, image2)
    cv2.imshow("Subtracted Image", subtracted_image)


adding(first_image, second_image)
subtracting(first_image, second_image)
close()

