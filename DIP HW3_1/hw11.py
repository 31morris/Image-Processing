import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the input image (grayscale)
image_path = 'aerial_view.tif'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 2: Compute the histogram
hist, bins = np.histogram(image.flatten(), 256, [0, 256])

# Step 3: Plot the image and its histogram
plt.figure(figsize=(6, 8))

# Plot the image
plt.subplot(2, 1, 1)
plt.imshow(image, cmap='gray')
plt.title('Aerial View Image')

# Plot the histogram
plt.subplot(2, 1, 2)
plt.hist(image.flatten(), 256, [0, 256], color='b')
plt.title('Histogram of Aerial View Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

# Display both
plt.tight_layout()
plt.show()
