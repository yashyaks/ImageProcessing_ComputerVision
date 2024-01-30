import cv2

original_image = cv2.imread('images/test.jpg')

def cropping(image):
    h, w, channels = original_image.shape
    # Calculate the starting coordinates for cropping the central part
    crop_height, crop_width = 800, 500
    x = max(0, (w - crop_width) // 2)
    y = max(0, (h - crop_height) // 2)
    cropped_image = original_image[y:y+crop_height, x:x+crop_width]
    cv2.imshow("Original Image", image)
    cv2.imshow("Cropped Image", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cropping(original_image)