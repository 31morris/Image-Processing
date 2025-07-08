import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the image
image = cv2.imread('astronaut-interference.tif', cv2.IMREAD_GRAYSCALE)

# Perform Fourier transform to convert the image to the frequency domain
f_transform = np.fft.fft2(image)
f_transform_shifted = np.fft.fftshift(f_transform)
magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)

# Set filter parameters
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
D0 = 35  # Cutoff frequency for the notch filter
W = 11   # Width of the filter

# Create a 2D filter to remove specific frequencies
notch_filter = np.ones((rows, cols), np.uint8)

# Set filter to zero in the frequency domain for specific frequency regions
notch_filter[crow-D0-W:crow-D0+W, ccol-D0-W:ccol-D0+W] = 0
notch_filter[crow+D0-W:crow+D0+W, ccol+D0-W:ccol+D0+W] = 0

# Apply the filter in the frequency domain
filtered_f_transform_shifted = f_transform_shifted * notch_filter

# Inverse Fourier transform back to the spatial domain
filtered_image = np.fft.ifftshift(filtered_f_transform_shifted)
filtered_image = np.fft.ifft2(filtered_image)
filtered_image = np.abs(filtered_image).astype(np.uint8)

# Calculate the magnitude spectrum of the filtered image
filtered_magnitude_spectrum = np.log(np.abs(filtered_f_transform_shifted) + 1)

# Display the original image and the processed results
plt.figure(figsize=(18, 8))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

# Display the filtered image
plt.subplot(1, 2, 2)
plt.title("Filtered Image (Notch Filter)")
plt.imshow(filtered_image, cmap='gray')

plt.show()
