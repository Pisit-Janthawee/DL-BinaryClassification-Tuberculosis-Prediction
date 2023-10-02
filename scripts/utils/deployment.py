import gradio as gr
import tensorflow as tf
import numpy as np
from lime.lime_image import LimeImageExplainer
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

import os
import cv2
import random

from skimage.filters import threshold_otsu
from skimage.exposure import equalize_adapthist
from skimage.color import label2rgb
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import square, closing

from skimage.segmentation import slic
from scripts.utils.pipeline import Pipeline


def lime_explainer_plot(X, estimator, prediction):
    X_resized = tf.image.resize(X[0], (128, 128)).numpy()

    rgb_resized_image = cv2.cvtColor(X_resized, cv2.COLOR_GRAY2RGB)

    explainer = LimeImageExplainer()

    def model_pred_fn(images):
        # Convert RGB to GrayScale, Because the model is trained grayscale
        images = images.mean(axis=-1, keepdims=True)
        y_hat = estimator.predict(images, verbose=0)
        return y_hat

    def segment_fn(image):
        return slic(image, n_segments=60, compactness=10, sigma=1)

    explanation = explainer.explain_instance(
        rgb_resized_image,
        model_pred_fn,
        top_labels=1,
        hide_color=0,
        num_samples=222,
        segmentation_fn=segment_fn
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True if prediction == 1 else False,
        num_features=5,
        hide_rest=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    image_label_overlay, label_image = lung_segmentation(
        X_resized.reshape((128, 128)))

    axes[0].imshow(X_resized, cmap='gray')
    actual_title = r'$\bf{' + 'Input' + '}$'

    axes[0].set_title(actual_title)
    axes[1].imshow(X_resized, cmap='gray')
    axes[1].imshow(mark_boundaries(
        image_label_overlay, mask))

    axes[1].set_title(
        r'$\bf{' + "Prediction" + '}$' + f'\nThe region has been classified as the positive class.')

    plt.suptitle(f'Image-Explanation', fontsize=16, fontweight='bold')

    plt.tight_layout()
    # Show the figure
    image_path = "lime_exp.png"

    plt.savefig(image_path, format="png")
    plt.show()
    return image_path


def predict_disease(image):
    pipeline = Pipeline()

    # Pre-Processing
    image_processed = pipeline.pre_processing(X=image)
    # Prediction
    model = tf.keras.models.load_model('artifacts/cnn_best_artifact.h5')
    # Probability
    probability = model.predict(image_processed)
    # Handling output
    prediction = (probability > 0.5).astype(int).flatten()

    # Return single probabilities, not arrays
    no_disease_probability = 1-float(probability[0])
    disease_probability = float(probability[0])
    output = {'No Disease': no_disease_probability,
              'Disease': disease_probability}
    print('after')
    image_path = lime_explainer_plot(
        X=image_processed, estimator=model, prediction=prediction)

    return output, image_path


def sampling_example_image():
    # Define the counts of Tuberculosis and Normal images
    num_tuberculosis = 700
    num_normal = 3500

    # Define the paths to Tuberculosis and Normal image directories
    tuberculosis_dir = r"D:\Repo\DS\Classification\BinaryClassification\Tuberculosis\repositorys\rawdata\Tuberculosis"
    normal_dir = r"D:\Repo\DS\Classification\BinaryClassification\Tuberculosis\repositorys\rawdata\Normal"

    # Get a random sample of Tuberculosis and Normal image file paths
    tuberculosis_images = random.sample(
        os.listdir(tuberculosis_dir), num_tuberculosis)
    normal_images = random.sample(os.listdir(normal_dir), num_normal)

    # Select a smaller random subset of images
    # Choose the smaller count as subset size
    subset_size = min(num_tuberculosis, num_normal)
    subset_tuberculosis = random.sample(tuberculosis_images, subset_size)
    subset_normal = random.sample(normal_images, subset_size)

    # Combine the two lists of image paths
    example_images = [os.path.join(tuberculosis_dir, img) for img in subset_tuberculosis] + \
        [os.path.join(normal_dir, img) for img in subset_normal]

    # Shuffle the combined list to ensure randomness
    random.shuffle(example_images)
    return example_images


def lung_segmentation(img):
    # Scale pixel values to the range of -1 to 1
    img = img.astype(np.float32) / 255.0

    # Even out the contrast with CLAHE
    img = equalize_adapthist(img, kernel_size=None,
                             clip_limit=0.01, nbins=256)

    # Make a binary threshold mask and apply it to the image
    thresh = threshold_otsu(image=img, nbins=256, hist=None)
    thresh = img > thresh
    bw = closing(img > thresh, square(1))

    # clean up the borders
    cleared = clear_border(bw)

    label_image = label(cleared)
    image_label_overlay = label2rgb(
        label_image,
        image=img,
        bg_label=0,
        bg_color=(0, 0, 0))
    return image_label_overlay, label_image


def lung_segmentation_for_lime(img):
    # Scale pixel values to the range of -1 to 1
    img = img.astype(np.float32) / 255.0

    # Even out the contrast with CLAHE
    img = equalize_adapthist(img, kernel_size=None,
                             clip_limit=0.01, nbins=256)

    # Make a binary threshold mask and apply it to the image
    thresh = threshold_otsu(image=img, nbins=256, hist=None)
    thresh = img > thresh
    bw = closing(img > thresh, square(1))

    # clean up the borders
    cleared = clear_border(bw)

    label_image = label(cleared)
 
    return label_image
