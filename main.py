# main.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import apply_fourier_transform, apply_filter


def main():
    # Load an image
    image_path = 'tiger.jpg'  # Replace with your image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Unable to load image.")
        return

    # Display the original image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Apply Fourier Transform
    magnitude_spectrum = apply_fourier_transform(image)

    # Display the magnitude spectrum
    plt.subplot(1, 3, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Fourier Magnitude Spectrum')
    plt.axis('off')

    # Apply a filter (e.g., Gaussian filter)
    filtered_image = apply_filter(image, filter_type='gaussian')

    # Display the filtered image
    plt.subplot(1, 3, 3)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Filtered Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
