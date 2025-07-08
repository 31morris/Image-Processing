import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the input image and convert to grayscale
image_path = 'aerial_view.tif'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 2: Calculate the constant c
# We assume the intensity values are in the range [0, 255]
z = np.arange(256)
target_pdf = z ** 0.4
c = 1 / np.sum(target_pdf)
target_pdf_normalized = c * target_pdf

# Step 3: Calculate the cumulative distribution function (CDF) for the target distribution
target_cdf = np.cumsum(target_pdf_normalized)
target_cdf_normalized = target_cdf / target_cdf[-1]  # Normalize to range [0, 1]

# Step 4: Compute the histogram and CDF of the input image
input_hist, bins = np.histogram(image.flatten(), 256, [0, 256])
input_cdf = input_hist.cumsum()
input_cdf_normalized = input_cdf / input_cdf[-1]

# Step 5: Perform histogram matching by finding the best mapping
mapping = np.zeros(256, dtype=np.uint8)
for i in range(256):
    diff = np.abs(target_cdf_normalized - input_cdf_normalized[i])
    mapping[i] = np.argmin(diff)

# Step 6: Apply the mapping to the original image to get the matched image
matched_image = mapping[image]

# Step 7: Plot the original and matched histograms
plt.figure(figsize=(12, 6))

# Original histogram
plt.subplot(2, 2, 1)
plt.hist(image.flatten(), 256, [0, 256], color='r')
plt.title('Original Image Histogram')

# Matched histogram (now blue instead of green)
plt.subplot(2, 2, 2)
plt.hist(matched_image.flatten(), 256, [0, 256], color='b')  # Changed to blue
plt.title('Matched Image Histogram')

# Step 8: Display the original and matched images
plt.subplot(2, 2, 3)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 4)
plt.imshow(matched_image, cmap='gray')
plt.title('Matched Image')

plt.tight_layout()
plt.show()
