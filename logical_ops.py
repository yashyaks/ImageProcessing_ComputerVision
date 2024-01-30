import cv2

def logical_and(image1, image2):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    result = cv2.bitwise_and(img1, img2)

    return result

def logical_or(image1, image2):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    result = cv2.bitwise_or(img1, img2)

    return result

def logical_not(image):
    img = cv2.imread(image)

    result = cv2.bitwise_not(img)

    return result

def logical_xor(image1, image2):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    result = cv2.bitwise_xor(img1, img2)

    return result

image1_path = 'images\drake_1.jpeg'
image2_path = 'images\drake_2.jpeg'

result_and = logical_and(image1_path, image2_path)
cv2.imshow('AND', result_and)

result_or = logical_or(image1_path, image2_path)
cv2.imshow('OR', result_or)

result_not = logical_not(image1_path)
cv2.imshow('NOT', result_not)

result_xor = logical_xor(image1_path, image2_path)
cv2.imshow('XOR', result_xor)

cv2.waitKey(0)
cv2.destroyAllWindows()
