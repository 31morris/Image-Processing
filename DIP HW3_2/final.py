import cv2
import numpy as np

# Define constants
k_const = [0, 0.3, 0, 0.04]

def compute_stats(img, size):
    """ 
    Calculate mean and standard deviation of the image 
    over a defined kernel size.
    """
    if size == 0:
        mean, std_dev = cv2.meanStdDev(img)
    else:
        kernel = np.ones((size, size), np.float32) / (size**2)
        mean = cv2.filter2D(img.astype(np.float32), -1, kernel)
        variance = cv2.filter2D((img.astype(np.float32) - mean)**2, -1, kernel)
        std_dev = np.sqrt(variance)
    return mean, std_dev

def calc_boundaries(a, b):
    """
    Calculate the boundary limits based on constants.
    """
    lower_a, upper_a = k_const[0]*a, k_const[1]*a
    lower_b, upper_b = k_const[2]*b, k_const[3]*b
    return [lower_a, upper_a, lower_b, upper_b]

def adjust_histogram(img, local_stats, max_val, bounds):
    """
    Adjust the pixel values in the image based on local statistics.
    """
    rows, cols = img.shape

    for x in range(1, rows):
        for y in range(1, cols):
            region = img[x-1:x+2, y-1:y+2]
            region_max = np.max(region)

            if bounds[0] < local_stats[0][x, y] < bounds[1] and bounds[2] < local_stats[1][x-1, y-1] < bounds[3]:
                correction_factor = round(max_val / region_max)
                img[x, y] = round(correction_factor * img[x, y])

    return img

if __name__ == '__main__':
    # Read the image
    image = cv2.imread('hidden_object_2.jpg', cv2.IMREAD_GRAYSCALE)

    # Calculate local mean and standard deviation
    local_measures = compute_stats(image, 3)
    
    # Find the maximum value in the image
    max_pixel_val = np.max(image)
    
    # Calculate the overall mean and standard deviation
    overall_mean_std = compute_stats(image, 0)

    # Calculate boundary limits
    boundaries = calc_boundaries(overall_mean_std[0], overall_mean_std[1])

    # Adjust the histogram of the image
    adjusted_img = adjust_histogram(image, local_measures, max_pixel_val, boundaries)

    # Save the resulting image
    cv2.imwrite('adjusted_histogram_image.png', adjusted_img)

    # Display the image
    cv2.imshow('Histogram Adjusted Image', adjusted_img)
    cv2.waitKey(0)
    
    # Save the processed data
    np.save('adjusted_histogram_data.npy', adjusted_img)

    # Load the saved data
    loaded_img = np.load('adjusted_histogram_data.npy')

    # Perform local enhancement
    clahe_processor = cv2.createCLAHE()
    enhanced_img = clahe_processor.apply(loaded_img)

    # Display the enhanced image
    cv2.imshow('local enhancement Image', enhanced_img)
    cv2.waitKey(0)
