import os
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian, convolve2d
import matplotlib.pyplot as plt
import cv2 
import math

def motion_process(image_size, motion_angle, degree=15):
    PSF = np.zeros(image_size)
    center_position=(image_size[0]-1)/2
    slope_tan=math.tan(motion_angle*math.pi/180)
    slope_cot=1/slope_tan
    if slope_tan<=1:
        for i in range(degree):
            offset=round(i*slope_tan)   
            PSF[int(center_position+offset),int(center_position-offset)]=1
        return PSF / PSF.sum() 
    else:
        for i in range(degree):
            offset=round(i*slope_cot)
            PSF[int(center_position-offset),int(center_position+offset)]=1
        return PSF / PSF.sum()

def get_motion_dsf(image_size, motion_angle, motion_dis):
   
    PSF = np.zeros(image_size) 
    x_center = (image_size[0] - 1) / 2
    y_center = (image_size[1] - 1) / 2
 
    sin_val = np.sin(motion_angle * np.pi / 180)
    cos_val = np.cos(motion_angle * np.pi / 180)
 
    for i in range(motion_dis):
        x_offset = round(sin_val * i)
        y_offset = round(cos_val * i)
        PSF[int(x_center - x_offset), int(y_center + y_offset)] = 1

    # normalized
    return PSF / PSF.sum()  
    
def make_blurred(input, PSF, eps):
    input_fft = np.fft.fft2(input)              
    PSF_fft = np.fft.fft2(PSF)+ eps             
    blurred = np.fft.ifft2(input_fft * PSF_fft) 
    blurred = np.abs(np.fft.fftshift(blurred))
    return blurred
 
def inverse_filter(input, PSF, eps): 
   
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF) + eps           
    result = np.fft.ifft2(input_fft / PSF_fft)
    result = np.abs(np.fft.fftshift(result))
    return result
 
def wiener_filter(input, PSF, eps, K=0.01):
   
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft)**2 + K)
    result = np.fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(np.fft.fftshift(result))
    return result

if __name__ == "__main__":
    
    image = cv2.imread('book-cover-blurred.tif', 0)
    
    PSF = get_motion_dsf(image.shape[:2], -50, 100)
    blurred = make_blurred(image, PSF, 1e-3)

    plt.figure(figsize=(18, 8))
    plt.subplot(131), plt.imshow(blurred, 'gray'), plt.title("Original Image")
    plt.xticks([]), plt.yticks([])

    # Inverse Filtering
    result = inverse_filter(blurred, PSF, 1e-3)   
    plt.subplot(132), plt.imshow(result, 'gray'), plt.title("inverse deblurred")
    plt.xticks([]), plt.yticks([])

    # Wiener FIltering
    result = wiener_filter(blurred, PSF, 1e-3, 0.000001)     
    plt.subplot(133), plt.imshow(result, 'gray'), plt.title("wiener deblurred(k=0.000001)")
    plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()