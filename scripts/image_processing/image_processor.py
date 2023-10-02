import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage import img_as_float


class ImageEnhancer:
    def __init__(self, image):
        self.image = image

    def contrast_stretching(self):
        p2, p98 = np.percentile(self.image, (2, 98))
        stretched_img = exposure.rescale_intensity(
            self.image, in_range=(p2, p98))
        return stretched_img

    def equalization(self):
        eq_img = exposure.equalize_hist(self.image)
        return eq_img

    def adaptive_equalization(self, clip_limit, batch_size=None):
        if batch_size is None:
            # Normalize the input image to the range of -1 to 1 and perform adaptive histogram equalization
            X_normalized = (self.image - np.min(self.image)) / \
                (np.max(self.image) - np.min(self.image))
            X_normalized = (X_normalized * 2) - 1  # Scale to -1 to 1
            adapteq_img = exposure.equalize_adapthist(
                X_normalized, clip_limit=clip_limit)
        else:
            # Process data in batches
            num_batches = int(np.ceil(len(self.image) / batch_size))
            batch_results = []

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(self.image))

                # Normalize the input image to the range of -1 to 1 and perform adaptive histogram equalization for each batch
                X_normalized = (self.image[start_idx:end_idx] - np.min(self.image[start_idx:end_idx])) / \
                    (np.max(self.image[start_idx:end_idx]) -
                     np.min(self.image[start_idx:end_idx]))
                X_normalized = (X_normalized * 2) - 1  # Scale to -1 to 1
                adapteq_img_batch = exposure.equalize_adapthist(
                    X_normalized, clip_limit=clip_limit)
                batch_results.append(adapteq_img_batch)

            # Concatenate the batch results
            adapteq_img = np.concatenate(batch_results, axis=0)

        return adapteq_img

    def img_and_hist(self, image, axes, bins=100):
        '''
        Plot an image along with its histogram and cumulative histogram.
        '''
        image = img_as_float(image)
        ax_img, ax_hist = axes
        ax_cdf = ax_hist.twinx()

        # Display image
        ax_img.imshow(image, cmap=plt.cm.gray)
        ax_img.set_axis_off()

        # Display histogram
        ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
        ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax_hist.set_xlabel('Pixel intensity')
        # ax_hist.set_xlim(0, 1)
        ax_hist.set_yticks([])

        # Display cumulative distribution
        img_cdf, bins = exposure.cumulative_distribution(image, bins)
        ax_cdf.plot(bins, img_cdf, 'r')
        ax_cdf.set_yticks([])

        return ax_img, ax_hist, ax_cdf
    def display_images(self):
        img_rescale = self.contrast_stretching()
        img_eq = self.equalization()
        img_adapteq = self.adaptive_equalization(0.03)

        methods = ['Low contrast image', 'Contrast stretching',
                   'Histogram equalization', 'Adaptive equalization']
        images = [self.image, img_rescale, img_eq, img_adapteq]

        fig, axes = plt.subplots(2, 4, figsize=(20, 8))

        for i, (method, image) in enumerate(zip(methods, images)):
            ax_img, ax_hist, ax_cdf = self.img_and_hist(image, axes[:, i], bins=555)

            mean_value = np.mean(image)
            std_value = np.std(image)
            min_value = np.min(image)
            max_value = np.max(image)

            ax_img.set_title(
                r'$\bf{' + f'{method}' + '}$' +
                f'\nMean: {mean_value: .2f}, Std: {std_value: .2f}, Min: {min_value: .2f}, Max: {max_value: .2f}',
                fontsize=12)

            y_min, y_max = ax_hist.get_ylim()
            ax_hist.set_title('Distribution of pixel intensities',
                              fontsize=12, fontweight='bold')
            ax_hist.set_ylabel('Number of pixels')
            ax_hist.set_yticks(np.linspace(0, y_max, 5))

            ax_cdf.set_ylabel('Fraction of total intensity')
            ax_cdf.set_yticks(np.linspace(0, 1, 5))

        plt.suptitle('Summary image enhancement techniques',
                     fontsize=16, fontweight='bold')
        # prevent overlap of y-axis labels
        fig.tight_layout()
        plt.show()

# Usage example:
# enhancer = ImageEnhancer(your_image)
# enhancer.display_images()
