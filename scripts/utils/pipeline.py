import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

from skimage import exposure


class Pipeline():
    def __init__(self):
        pass

    def contrast_stretching(self, X):
        p2, p98 = np.percentile(X, (2, 98))
        stretched_img = exposure.rescale_intensity(X, in_range=(p2, p98))
        return stretched_img

    def pre_processing(self, X):

        grayscale_image = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)

        # Resize to (128, 128)
        resized_image = cv2.resize(grayscale_image, (128, 128))

        # Expand dimensions to (128, 128, 1)
        new_image = resized_image[:, :, np.newaxis]
        # IF_WANT_TO_SEE_IMAGE_AFTER_PROCESSING
        fig, axes = plt.subplots(1, 2, figsize=(20, 5))
        plt.suptitle(f'Image shape', fontsize=16, fontweight='bold')

        fig.set_facecolor('white')

        # Display the original image parameters in the first column
        axes[0].imshow(X[:, :, 0], cmap='gray')
        axes[0].set_title(f'Original Image\n{X.shape}')

        # Apply contrast stretching
        stretched_img = self.contrast_stretching(X=new_image)

        # Display the stretched image in the second column
        axes[1].imshow(stretched_img[:, :, 0], cmap='gray')
        axes[1].set_title(f'after Contrast Stretching\n{stretched_img.shape}')

        image_path = "deploy_input_image.png"
        plt.savefig(image_path, format="png")
        #return(1,128,128,1)
        return stretched_img[np.newaxis, :, :, :]
