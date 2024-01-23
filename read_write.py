# imread() helps us read an image
# imshow() displays an image in a window
# imwrite() writes an image into the file directory

import cv2
img = cv2.imread('images/test.jpg')
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()