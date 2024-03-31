import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion MNIST dataset
(_, _), (x_test, _) = fashion_mnist.load_data()

# Select a random image from the dataset
random_index = np.random.randint(0, len(x_test))
image = x_test[random_index]

# Apply Fourier Transform
f_transform = np.fft.fft2(image)
f_shift = np.fft.fftshift(f_transform)

# Ideal Low Pass Filter
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
d = 30  # Cut-off frequency
mask = np.zeros((rows, cols), np.uint8)
mask[crow - d:crow + d, ccol - d:ccol + d] = 1

f_shift_low = f_shift * mask
f_ishift_low = np.fft.ifftshift(f_shift_low)
img_back_low = np.fft.ifft2(f_ishift_low)
img_back_low = np.abs(img_back_low)

# Ideal High Pass Filter
mask_high = np.ones((rows, cols), np.uint8)
mask_high[crow - d:crow + d, ccol - d:ccol + d] = 0

f_shift_high = f_shift * mask_high
f_ishift_high = np.fft.ifftshift(f_shift_high)
img_back_high = np.fft.ifft2(f_ishift_high)
img_back_high = np.abs(img_back_high)

# Gaussian Low Pass Filter
mask_gaussian_low = cv2.GaussianBlur(image, (5, 5), 0)

# Gaussian High Pass Filter
mask_gaussian_high = cv2.subtract(image, mask_gaussian_low)

# Plotting the images
plt.figure(figsize=(10, 10))

plt.subplot(3, 4, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(3, 4, 2)
plt.title('Ideal Low Pass Filtered')
plt.imshow(img_back_low, cmap='gray')

plt.subplot(3, 4, 3)
plt.title('Ideal High Pass Filtered')
plt.imshow(img_back_high, cmap='gray')

plt.subplot(3, 4, 4)
plt.title('Gaussian Low Pass Filtered')
plt.imshow(mask_gaussian_low, cmap='gray')

plt.subplot(3, 4, 5)
plt.title('Gaussian High Pass Filtered')
plt.imshow(mask_gaussian_high, cmap='gray')

plt.show()