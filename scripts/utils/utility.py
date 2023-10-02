import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage import img_as_float
from skimage import exposure

from sklearn.model_selection import train_test_split


def plot_image_distribution(X, title):
    # Create a subplot with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the distribution of the first column in the first axis
    sns.distplot(X[:, 0], bins=20, color="black", ax=axes[0])
    axes[0].set_title("Height Distribution")

    # Plot the distribution of the second column in the second axis
    sns.distplot(X[:, 1], bins=20, color="black", ax=axes[1])
    axes[1].set_title("Width Distribution")
    # Set the overall title for the subplots
    fig.suptitle(title, fontsize=12)

    plt.show()


def plot_image_scatter(X, title):
    # Create a scatter plot for the specified column vs. index
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, 1], color="black", alpha=0.1)
    plt.title(title)
    plt.xlabel("width")
    plt.ylabel("height")
    plt.grid(True)
    plt.show()

def plot_gray_scale_histogram(images, titles, bins=100):
    '''
    Plot Gray Scale Histograms of Images.

    Parameters:
        - images (list): List of grayscale images to plot histograms for.
        - titles (list): List of titles for each histogram.
        - bins (int, optional): Number of bins for the histogram. Default is 100.

    Returns:
        None

    This function generates histograms for a list of grayscale images and displays them side by side. Each histogram is accompanied by its respective title.

    The function does not return any values; it displays the histogram plots directly.
    '''
    # Display results
    fig, axes = plt.subplots(2, len(images), figsize=(20, 8))

    for i, (title, image) in enumerate(zip(titles, images)):
        ax_img, ax_hist, ax_cdf, random_index = img_and_hist(
            image, axes[:, i], bins)

        mean_value = np.mean(image)
        std_value = np.std(image)
        min_value = np.min(image)
        max_value = np.max(image)

        ax_img.set_title('Random image of '+r'$\bf{' + f'{title}'+'}$' +
                         f'\nMean: {mean_value: .2f}, Std: {std_value: .2f}, Min: {min_value: .2f}, Max: {max_value: .2f}', fontsize=16)
        ax_img.text(0.5, -0.15, f'Image Index: {random_index}\n(Display random image)', transform=ax_img.transAxes,
                    fontsize=10, ha='center')

        y_min, y_max = ax_hist.get_ylim()
        ax_hist.set_title(
            'Distribution of pixel intensities of'+r'$\bf{' + f'{title}'+'}$', fontsize=16)
        ax_hist.set_ylabel('Number of pixels')
        ax_hist.set_yticks(np.linspace(0, y_max, 5))

        ax_cdf.set_ylabel('Fraction of total intensity')
        ax_cdf.set_yticks(np.linspace(0, 1, 5))

    plt.suptitle('Gray scale Histogram: Distribution of intensity pixel',
                 fontsize=16, fontweight='bold')
    # Prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()


def img_and_hist(image_data, axes, bins):
    '''
    Plot an image along with its histogram and cumulative histogram.

    Parameters:
        - image_data (ndarray): Grayscale image data as a numpy array.
        - axes (list): List of axes for displaying the image, histogram, and cumulative histogram.
        - bins (int): Number of bins for the histogram.

    Returns:
        - ax_img, ax_hist, ax_cdf: Axes objects for image, histogram, and cumulative histogram.

    This function displays an image along with its histogram and cumulative histogram. It takes the grayscale image data, a list of axes for plotting, and the number of bins for the histogram.

    The function returns the axes objects for the image, histogram, and cumulative histogram.
    '''
    bins = bins if bins else 100
    '''
    Plot an image along with its histogram and cumulative histogram.
    '''
    image = img_as_float(image_data)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    random_index = np.random.randint(0, len(image_data))

    # Display image
    ax_img.imshow(image[random_index], cmap=plt.cm.gray)
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

    return ax_img, ax_hist, ax_cdf, random_index


def plot_scree_cumulative(explain_ratio, where_n_component, plot_type='plotly'):
    """
    Create a Scree plot with cumulative variation.

    Parameters:
    - explain_ratio (ndarray): Explained variance ratios for each component.
    - where_n_component (int): Number of components to mark as the "elbow" point.
    - plot_type (str): 'plotly' or 'seaborn' to specify the plotting library.

    Returns:
    - fig (object): Plotly figure if plot_type is 'plotly', otherwise None.
    """

    explain_ratio_cum = explain_ratio.cumsum()

    if plot_type == 'plotly':
        fig = go.Figure()

        # Add cumulative and individual explained variance ratios
        fig.add_trace(go.Scatter(x=list(range(1, len(explain_ratio) + 1)), y=explain_ratio_cum,
                                 mode='lines', name='Cumulative', line=dict(color='firebrick')))
        fig.add_trace(go.Scatter(x=list(range(1, len(explain_ratio) + 1)), y=explain_ratio,
                                 mode='lines', name='Individual', line=dict(color='royalblue', dash='dash')))

        # Adding values to the plot
        for x, ex_ratio, ex_ratio_cum in zip(range(1, len(explain_ratio) + 1),
                                             explain_ratio,
                                             explain_ratio_cum):
            if x <= 100 and x % 20 == 0:
                ex_ratio_label = f'{ex_ratio * 100:.0f}%'
                fig.add_annotation(x=x, y=ex_ratio, text=ex_ratio_label,
                                   xshift=5, yshift=5, showarrow=False, font=dict(size=10))
                ex_ratio_cum_label = f'{ex_ratio_cum * 100:.0f}%'
                fig.add_annotation(x=x, y=ex_ratio_cum, text=ex_ratio_cum_label,
                                   xshift=5, yshift=5, showarrow=False, font=dict(size=10))
            else:
                if x % 500 == 0:
                    ex_ratio_label = f'{ex_ratio * 100:.2f}%'
                    fig.add_annotation(x=x, y=ex_ratio, text=ex_ratio_label,
                                       xshift=5, yshift=5, showarrow=False, font=dict(size=10))
                    ex_ratio_cum_label = f'{ex_ratio_cum * 100:.2f}%'
                    fig.add_annotation(x=x, y=ex_ratio_cum, text=ex_ratio_cum_label,
                                       xshift=5, yshift=5, showarrow=False, font=dict(size=10))

        # Mark the "elbow" point with lines and annotation
        fig.add_shape(type="line", x0=where_n_component, y0=0, x1=where_n_component, y1=1,
                      line=dict(color="brown", width=1, dash="dash"))
        fig.add_shape(type="line", x0=1, y0=0.01, x1=len(explain_ratio), y1=0.05,
                      line=dict(color="brown", width=1, dash="dash"))
        fig.add_annotation(x=where_n_component, y=0.2, text=f'Variance explained = {(explain_ratio_cum[where_n_component]*100).round(2)}%',
                           showarrow=False, font=dict(size=12), xshift=5, yshift=5)

        # Set x-axis limits
        fig.update_xaxes(range=[1, len(explain_ratio)])

        # Update layout
        fig.update_layout(title_text=f"<b>Scree plot</b> with cumulative variation plot <br><b>criterion</b>: looks for the “elbow” in the curve<br><b>choose:</b> n_components = {where_n_component}",
                          xaxis_title="Number of components",
                          yaxis_title="Variance explained",
                          legend=dict(x=0.02, y=0.98, bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='rgba(0, 0, 0, 0.5)'))

        return fig
    elif plot_type == 'seaborn':
        plt.figure(figsize=(20, 8))
        with plt.style.context('seaborn'):
            plt.title(
                r'$\bf{'+'Scree plot and cumulative variation plot'+'}$'+'\nFinding Optimum Number of Principle Component')
            plt.xlabel('Number of components')
            plt.ylabel('Variance explained')
            plt.plot(range(1, explain_ratio.shape[0] + 1),
                     explain_ratio_cum, c='firebrick', label='Cumulative')
            plt.plot(range(1, explain_ratio.shape[0] + 1), explain_ratio,
                     c='royalblue', linestyle='--', label='Individual')

            # Adding values to plot
            for x, ex_ratio, ex_ratio_cum in zip(range(1, explain_ratio.shape[0] + 1),
                                                 explain_ratio,
                                                 explain_ratio_cum):
                if x < 100 and x % 20 == 0:
                    ex_ratio_label = f'{ex_ratio * 100:.0f}%'

                    plt.annotate(ex_ratio_label, (x, ex_ratio), textcoords='offset points',
                                xytext=(5, 5), ha='center')
                    ex_ratio_cum_label = f'{ex_ratio_cum * 100:.0f}%'
                    plt.annotate(ex_ratio_cum_label, (x, ex_ratio_cum), textcoords='offset points',
                                xytext=(5, 5), ha='center')
                else:
                    if x % 500 == 0:
                        ex_ratio_label = f'{ex_ratio * 100:.2f}%'

                        plt.annotate(ex_ratio_label, (x, ex_ratio), textcoords='offset points',
                                    xytext=(5, 5), ha='center')
                        ex_ratio_cum_label = f'{ex_ratio_cum * 100:.2f}%'
                        plt.annotate(ex_ratio_cum_label, (x, ex_ratio_cum), textcoords='offset points',
                                    xytext=(5, 5), ha='center')

    
def plot_class_distribution(X, y, title, classes, train_percent=0.6, val_percent=0.2, test_percent=0.2):
    '''
    :Parameters:
    - X (numpy array): The input feature matrix of shape (num_examples, num_features).
    - y (numpy array): The target labels of shape (num_examples,).
    - title (str): The title for the entire plot.
    - classes (list): A list of class labels, e.g., ['Normal', 'Tuberculosis'].
    - train_percent (float): Percentage of data for training set.
    - val_percent (float): Percentage of data for validation set.
    - test_percent (float): Percentage of data for test set.

    :Returns:
    None (Displays the plot).
    '''
    assert train_percent + val_percent + \
        test_percent == 1.0, "Sum of train_percent, val_percent, and test_percent should be 1.0"

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_percent+test_percent, random_state=42, stratify=y)

    test_size = test_percent / (val_percent + test_percent)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size, random_state=42, stratify=y_temp)

    # Create a subplot with 3 columns and 1 row
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # List of subset names
    subset_names = ['Train', 'Test', 'Validation']
    n = len(X)

    # Iterate over subsets
    for i, subset in enumerate([(X_train, y_train), (X_test, y_test), (X_val, y_val)]):
        X_subset, y_subset = subset
        n_subset = len(y_subset)
        # Get the class counts
        class_counts = np.bincount(y_subset)

        # Plot histogram for current subset
        axs[i].bar(classes, class_counts, color='#AA99FF')
        subtitle = r'$\bf{' + subset_names[i] + \
            '}$' + f' {int(n_subset/n*100)} %'
        axs[i].set_title(
            subtitle + f'\n Size = {X_subset.shape[0]}', fontsize=18)
        axs[i].set_xlabel('Class')
        axs[i].set_ylabel('Number of examples')

        # Add labels to the bars
        for j, count in enumerate(class_counts):
            axs[i].text(j, count, str(count), ha='center',
                        va='bottom', fontsize=12)

    class_counts = np.bincount(y)
    class_balance_text = ' | '.join(
        [f'{class_label}: {count}' for class_label, count in zip(classes, class_counts)])
    plt.suptitle(f'{title}' + f'\n Training examples (X) = {X.shape[0]}' +
                 f'\n Class balance = {class_balance_text}', fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.show()


def show_image(images, target, title, num_display=16, num_cols=4, cmap='gray', random_mode=False):
    '''
    :Parameters
        images (ndarray (n,)): Input data as a numpy array.
        target (ndarray (n,)): Target data as a numpy array.
        title (String): Title of the plot.
        num_display (int): Number of images to display. Default is 16.
        num_cols (int): Number of columns in the plot. Default is 4.
        cmap (str): Color map for displaying images. Default is 'gray'.
        random_mode (bool): If True, display images randomly. If False, display the first num_display images. Default is False.
    '''
    # Determine the number of rows based on the num_cols parameter
    n_cols = min(num_cols, num_display)
    n_rows = int(np.ceil(num_display / n_cols))

    n_images = min(num_display, len(images))
    if random_mode:
        random_indices = np.random.choice(
            len(images), num_display, replace=False)
    else:
        random_indices = np.arange(num_display)

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(20, 4*n_rows))
    for i, ax in enumerate(axes.flatten()):
        if i >= n_images:  # Check if the index exceeds the available number of images
            break
        # Incase (Did PCA)
        index = random_indices[i]
        if len(images.shape) == 2:
            image = images[index].reshape((128, 128)).astype(int)
        else:
            image = images[index]

        ax.imshow(image, cmap=cmap)
        ax.set_title("Target: {}".format(target[index]))

        # Add image index as text
        ax.text(0.5, -0.15, f'Image Index: {index}', transform=ax.transAxes,
                fontsize=10, ha='center')

    plt.suptitle(f"{title} (Displaying {num_display} Images)",
                 fontsize=16, fontweight='bold')

    fig.set_facecolor('white')
    plt.tight_layout()  # Added to ensure proper spacing between subplots
    return plt.show()


def compare_actual_and_predicted_xray(estimator, images, target, title, num_display=16, num_cols=4, random_mode=False):
    '''
    Compare Actual X-ray Images with Model Predictions.

    Parameters:
        estimator: Model used for predictions.
        images (ndarray): Input data as a numpy array.
        target (ndarray): Target data as a numpy array.
        title (str): Title of the plot.
        num_display (int, optional): Number of images to display. Default is 16.
        num_cols (int, optional): Number of columns in the plot. Default is 4.
        random_mode (bool, optional): If True, display images randomly. If False, display the first num_display images. Default is False.

    Returns:
        None

    This function generates a visual comparison between actual X-ray images and their corresponding model predictions. It displays a grid of images with labels to show whether the model's predictions match the actual target values. The grid is organized in rows and columns based on the specified parameters.

    The function does not return any values; it displays the comparison plot directly.
    '''

    # Check if the inputs are valid
    if images.shape[0] != len(target):
        print("Error: Number of images (does not match) the number of targets.")
        return

    if num_display > images.shape[0]:
        print("Error: add more images on images parameters ja ! ")
        return

    if not isinstance(num_cols, int) or num_cols <= 0:
        print("Error: num_cols (should be a) positive integer.")
        return
    
    if not isinstance(num_display, int) or num_display <= 0:
        print("Error: num_display (should be a) positive integer.")
        return

    # Determine the number of rows based on the num_cols parameter
    n_cols = min(num_cols, num_display)
    n_rows = int(np.ceil(num_display / n_cols))

    title = r'$\bf{' + "Actual-Image" + '}$' + " vs " + \
        r'$\bf{' + "Model-Prediction" + '}$'
    y_hat = estimator.predict(images)
    prediction = (y_hat > 0.5).astype(int).flatten()

    n_images = min(num_display, len(images))
    if random_mode:
        random_indices = np.random.choice(
            len(images), num_display, replace=False)
    else:
        random_indices = np.arange(num_display)

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(20, 4*n_rows))
    for i, ax in enumerate(axes.flatten()):
        if i >= n_images:  # Check if the index exceeds the available number of images
            break
        # Incase (Did PCA)
        index = random_indices[i]
        if len(images.shape) == 2:
            image = images[index].reshape((128, 128)).astype(int)
        else:
            image = images[index]
        actual_label = target[index]
        model_pred_label = prediction[index]
        model_prob = '{:.3f}'.format(float(y_hat[index]))
        ax.imshow(
            image, cmap='gray' if actual_label == model_pred_label else 'OrRd')
        ax.set_title(
            f"Actual: {actual_label},\nModel Prediction: {model_pred_label}\nProbability: {model_prob}")
    plt.suptitle(f"{title} (Displaying {num_display} Images)",
                 fontsize=16, fontweight='bold')

    fig.set_facecolor('white')
    plt.tight_layout()  # Added to ensure proper spacing between subplots
    plt.show()
