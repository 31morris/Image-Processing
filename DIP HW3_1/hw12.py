import cv2
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('aerial_view.tif', cv2.IMREAD_GRAYSCALE)

# Apply Histogram Equalization
equalized_image = cv2.equalizeHist(image)

# Plotting the original and equalized histograms
plt.figure(figsize=(12, 6))

# Original Image Histogram
plt.subplot(2, 2, 1)
plt.hist(image.ravel(), bins=256, range=(0, 256))
plt.title('Original Image Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

# Equalized Image Histogram
plt.subplot(2, 2, 2)
plt.hist(equalized_image.ravel(), bins=256, range=(0, 256))
plt.title('Equalized Image Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

# Displaying Original Image
plt.subplot(2, 2, 3)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Displaying Equalized Image
plt.subplot(2, 2, 4)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

# Show the plots
plt.tight_layout()
plt.show()
