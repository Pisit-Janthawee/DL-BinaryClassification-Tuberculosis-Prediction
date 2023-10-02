import matplotlib.pyplot as plt
import numpy as np
import random


def plot_history_comparison(name_list, color_map, historys, title, show_train=True):
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    best_epochs_accuracy = {}
    best_epochs_loss = {}
    best_val_accuracy = {}

    for name, history in zip(name_list, historys):
        epochs = np.arange(1, len(history.history['accuracy']) + 1)
        val_loss = history.history['val_loss']
        val_accuracy = history.history['val_accuracy']

        if show_train:
            train_loss = history.history['loss']
            train_accuracy = history.history['accuracy']

        line_color = color_map.get(name, 'black')
        name = name.capitalize()

        if show_train:
            axes[0].plot(
                epochs, train_loss, label=f'{name} (Train = {train_loss[-1]:.3f})', color=line_color, alpha=0.5)
        if not show_train:
            axes[0].plot(
                epochs, val_loss, label=f'{name} (Valid = {val_loss[-1]:.3f})', color=line_color, alpha=0.5)

        best_epoch_loss = np.argmin(val_loss) + 1
        axes[0].scatter(best_epoch_loss, val_loss[best_epoch_loss - 1],
                        marker='x', color=line_color, s=100)

        best_epochs_loss[name] = best_epoch_loss

        if show_train:
            axes[1].plot(epochs, train_accuracy,
                         label=f'{name} (Train = {train_accuracy[-1]:.3f})', color=line_color, alpha=0.5)
        if not show_train:
            axes[1].plot(epochs, val_accuracy,
                         label=f'{name} (Valid = {val_accuracy[-1]:.3f})', color=line_color, alpha=0.5)

        best_epoch_accuracy = np.argmax(val_accuracy) + 1
        axes[1].scatter(best_epoch_accuracy, val_accuracy[best_epoch_accuracy - 1],
                        marker='x', color=line_color, s=100)

        best_epochs_accuracy[name] = best_epoch_accuracy
        best_val_accuracy[name] = val_accuracy[best_epoch_accuracy - 1]

    axes[0].legend(loc='upper right', fontsize=12)
    axes[1].legend(loc='lower right', fontsize=12)

    axes[0].set_xlim([0.0, len(epochs)])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('Epoch', fontsize=15)
    axes[0].set_ylabel('Loss', fontsize=15)
    if show_train:
        axes[0].set_title('Training Loss', fontsize=20)
    else:
        axes[0].set_title('Validation Loss', fontsize=20)

    axes[1].set_xlim([0.0, len(epochs)])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Epoch', fontsize=15)
    axes[1].set_ylabel('Accuracy', fontsize=15)
    if show_train:
        axes[1].set_title('Training Accuracy', fontsize=20)
    else:
        axes[1].set_title('Validation Accuracy', fontsize=20)

    if show_train:
        performance_type = "Training"
    else:
        performance_type = "Validation"

    fig.suptitle(r'$\bf{' + title +
                 '}$' + f'\n Epoch = {len(epochs)} - {performance_type} Performance', fontsize=18)

    plt.tight_layout()
    plt.show()


def plot_accuracy_comparison(df, title):
    fig, ax = plt.subplots(figsize=(18, 9))
    legend_labels = []

    for index, row in df.iterrows():
        accuracy_train_list = row['accuracy_train']
        line, = plt.plot(accuracy_train_list, label=f'Model {row["Model"]}')
        config_dict = row["Hyperparameters Configuration"]
        config = " ".join(f"{value}" for key, value in config_dict.items())
        legend_labels.append(
            f'Model {row["Model"]} {config} ({max(accuracy_train_list):.3f})')

        best_epoch = accuracy_train_list.index(max(accuracy_train_list))
        best_accuracy = max(accuracy_train_list)

        random_offset_x = random.randint(-60, 60)
        random_offset_y = random.randint(-60, 60)

        plt.annotate(f'{best_accuracy:.3f}',
                     xy=(best_epoch, best_accuracy),
                     xytext=(random_offset_x, random_offset_y),
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->'))

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(
        title, fontsize=18)
    plt.legend(legend_labels)
    plt.show()
