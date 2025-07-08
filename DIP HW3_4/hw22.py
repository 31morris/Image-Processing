import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define Sobel filter
def sobel_filter(image):
    # Define Sobel kernels (standard Sobel kernels)
    sobel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32)  # Horizontal direction

    sobel_y = np.array([[1,  2,  1],
                         [0,  0,  0],
                         [-1, -2, -1]], dtype=np.float32)  # Vertical direction

    # Apply the filter
    grad_x = cv2.filter2D(image, cv2.CV_64F, sobel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, sobel_y)

    # Compute the gradient magnitude and normalize
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)  # Ensure result is within range
    return magnitude

# Define Laplacian filter (3x3 kernel)
def laplacian_filter(image):
    # Define Laplacian kernel (3x3 kernel)
    laplacian = np.array([[0,  1, 0],
                          [1, -4, 1],
                          [0,  1, 0]], dtype=np.float32)

    # Apply the filter
    laplacian_result = cv2.filter2D(image, cv2.CV_64F, laplacian)
    laplacian_result = np.clip(laplacian_result, 0, 255).astype(np.uint8)  # Ensure result is within range
    return laplacian_result

# Define high-boost method
def high_boost(image, alpha=1.0):
    # Use Gaussian blur for noise reduction
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Convert to grayscale
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

    # Apply Sobel and Laplacian filters
    sobel_edges = sobel_filter(gray_image)
    laplacian_edges = laplacian_filter(gray_image)

    # Combine Sobel and Laplacian edges
    edges = np.clip(sobel_edges + 0.5 * laplacian_edges, 0, 255).astype(np.uint8)

    # Convert edges to color
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # High-boost image, using the original color image as the base
    high_boost_image = cv2.addWeighted(image, 1 + alpha, edges_colored, -alpha, 0)
    return gray_image, sobel_edges, laplacian_edges, high_boost_image

# Main function
if __name__ == "__main__":
    # Load the image
    image = cv2.imread('fish.jpg')  # Replace with your image path

    if image is None:
        print("Unable to read the image, please check the file path.")
    else:
        # Apply high-boost method
        gray_image, sobel_edges, laplacian_edges, high_boost_image = high_boost(image)

        # Display the images
        plt.figure(figsize=(12, 8))

        # Original image
        plt.subplot(2, 2, 1)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert color channels
        plt.axis('off')

        # Sobel filter result
        plt.subplot(2, 2, 2)
        plt.title('Sobel Filter Result')
        plt.imshow(sobel_edges, cmap='gray')
        plt.axis('off')

        # Laplacian filter result
        plt.subplot(2, 2, 3)
        plt.title('Laplacian Filter Result')
        plt.imshow(laplacian_edges, cmap='gray')
        plt.axis('off')

        # High boost enhanced image
        plt.subplot(2, 2, 4)
        plt.title('High Boost Enhanced Image')
        plt.imshow(cv2.cvtColor(high_boost_image, cv2.COLOR_BGR2RGB))  # Ensure color channel conversion
        plt.axis('off')
        plt.tight_layout()
        plt.show()
