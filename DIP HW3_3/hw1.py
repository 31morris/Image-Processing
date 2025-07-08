import numpy as np
import cv2
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma=1):
    """
    Generate a Gaussian filter kernel
    size: The size of the kernel (should be an odd number)
    sigma: The standard deviation of the Gaussian function
    """
    # Create a grid of size x size
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)

    # Calculate the Gaussian function
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    # Normalize the kernel so that the sum of all values is 1
    kernel = kernel / np.sum(kernel)

    return kernel

# Set the kernel size and standard deviation
kernel_size = 255  # 255x255 kernel
sigma_value = 64

# Generate the Gaussian kernel
gaussian_kernel_result = gaussian_kernel(kernel_size, sigma_value)

# Read the image
image = cv2.imread('checkerboard1024-shaded.tif', cv2.IMREAD_GRAYSCALE)
# print(np.shape(image))
# cv2.imwrite('checkerboard.png', image)

# Apply the filter to the image
cvfilter = cv2.filter2D(image, -1, gaussian_kernel_result)
image2 = image / cvfilter

# Show the original image and the processed image
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title("Shaded Pattern")
plt.imshow(cvfilter, cmap='gray')
plt.subplot(1, 3, 3)
plt.title("Corrected Image")
plt.imshow(image2, cmap='gray')
plt.show()

# Save the processed images
cv2.imwrite('checkerboard_original.png', image)
cv2.imwrite('checkerboard_shaded.png', cvfilter)
cv2.imwrite('checkerboard_gaussian.png', image2)
