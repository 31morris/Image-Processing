import cv2
import numpy as np
import matplotlib.pyplot as plt

def notch_reject_filter(shape, d0=9, u_k=0, v_k=0):
    P, Q = shape
    H = np.ones((P, Q))
    for u in range(0, P):
        for v in range(0, Q):
            D_uv = np.sqrt((u - P / 2 + u_k) ** 2 + (v - Q / 2 + v_k) ** 2)
            D_muv = np.sqrt((u - P / 2 - u_k) ** 2 + (v - Q / 2 - v_k) ** 2)
            if D_uv <= d0 or D_muv <= d0:
                H[u, v] = 0.0
    return H

# Read the image
img = cv2.imread('car-moire-pattern.tif',0)

# Perform Fourier transform
f_transform_moire = np.fft.fft2(img)
f_shifted_moire = np.fft.fftshift(f_transform_moire)
magnitude_spectrum = 20 * np.log(np.abs(f_shifted_moire))

# Set parameters for multiple filters
img_shape = img.shape
H1 = notch_reject_filter(img_shape, 4, 38, 30)
H2 = notch_reject_filter(img_shape, 4, -42, 27)
H3 = notch_reject_filter(img_shape, 2, 80, 30)
H4 = notch_reject_filter(img_shape, 2, -82, 28)

# Combine the filters
NotchFilter = H1 * H2 * H3 * H4
NotchRejectCenter = f_shifted_moire * NotchFilter

# Inverse Fourier transform to restore the image
NotchReject = np.fft.ifftshift(NotchRejectCenter)
inverse_NotchReject = np.fft.ifft2(NotchReject)
Result = np.abs(inverse_NotchReject)

# Display results
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Filtered Image (Notch Filter)")
plt.imshow(Result, cmap='gray')

plt.show()
