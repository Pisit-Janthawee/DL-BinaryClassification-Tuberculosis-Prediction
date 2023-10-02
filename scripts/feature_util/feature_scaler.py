import cv2
import numpy as np

class ImageScaler:
    def __init__(self):
        pass

    def min_max(self, image):
        '''
        Perform Min-Max Scaling on an input image to scale the pixel values between 0 and 1.

        :Parameters:
        image (ndarray): Input image as a numpy array

        :Returns:
        ndarray: Scaled image with pixel values in the range [0, 1]
        '''
        # Use when?
        # - you want to scale pixel values to a range between 0 and 1.
        # - Preserve the original distribution's shape
        min_val = np.min(image)
        max_val = np.max(image)
        scaled_image = (image - min_val) / (max_val - min_val)

        return scaled_image

    def z_score(self, image):
        '''
        Perform Z-Score Scaling on an input image to standardize the pixel values.

        :Parameters:
        image (ndarray): Input image as a numpy array

        :Returns:
        ndarray: Scaled image with standardized pixel values (mean=0, std=1)
        '''
        # Use when you want to standardize pixel values to have a mean of 0 and a standard deviation of 1.
        mean_val = np.mean(image)
        std_val = np.std(image)
        scaled_image = (image - mean_val) / std_val

        return scaled_image

    def scale_to_range(self, image, range_min=0, range_max=255):
        '''
        Scale pixel values of an input image to a custom range.

        :Parameters:
        image (ndarray): Input image as a numpy array
        range_min (int): Minimum value for scaling (default: 0)
        range_max (int): Maximum value for scaling (default: 255)

        :Returns:
        ndarray: Scaled image with pixel values in the specified range
        '''
        # Use when you want to scale pixel values to a custom range (e.g., [0, 255]).
        min_val = np.min(image)
        max_val = np.max(image)
        scaled_image = ((image - min_val) / (max_val - min_val)) * (range_max - range_min) + range_min

        return scaled_image
