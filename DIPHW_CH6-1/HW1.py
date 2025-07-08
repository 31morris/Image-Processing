#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(file_path):
    frames = []
    image = cv2.imread(file_path)  
    if image is None:
        raise ValueError("Failed to load the image!")
    frames = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]  
    return frames

def compute_gradient(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def detect_edges(frames, threshold=50):
    edge_results = []
    for frame in frames:
        gradient_magnitude = compute_gradient(frame)
        _, edge_image = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)
        edge_results.append(edge_image)
    return edge_results

def visualize_results(frames, edge_results):
    for i, (original, edge) in enumerate(zip(frames, edge_results)):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(original)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title('Edge Detected Image')
        plt.imshow(edge, cmap='gray')
        plt.axis('off')
        plt.show()

def save_results(edge_results, file_path, output_prefix="edge_result"):
    output_prefix = f"{output_prefix}_for_{os.path.basename(file_path)}"
    for i, edge_image in enumerate(edge_results):
        output_path = f"{output_prefix}_frame_{i + 1}.png"
        cv2.imwrite(output_path, edge_image)
        print(f"Saved: {output_path}")

if __name__ == '__main__':
    file_path = "lenna-RGB.tif"  
    frames = load_image(file_path)  
    edge_results = detect_edges(frames, threshold=50)
    visualize_results(frames, edge_results)
    save_results(edge_results, file_path)
