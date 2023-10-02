import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_history(history, model, model_name, title, ref_name, config_param, training_time):
    '''
    :Parameters:
        - history (object): History object returned by the model.fit() method.
        - model (Object): CNN model
        - model_name (String): name of model
        - training_time (float): Time taken for training in seconds.
        - config_param (Dict): Hyperparameters configuration and optimizer parameters.
    '''

    # Get the index of the epoch with the highest validation accuracy
    best_epoch = np.argmax(history.history['val_accuracy'])
    best_train_epoch = np.argmax(history.history['accuracy'])

    # Create a 1x2 grid of subplots
    fig, axes = plt.subplots(1, 2, figsize=(24, 15))

    # Plot training and validation accuracy
    axes[0].plot(history.epoch, history.history['accuracy'],
                 label='Train Accuracy')
    axes[0].plot(history.epoch, history.history['val_accuracy'],
                 label='Validation Accuracy')
    axes[0].scatter(best_epoch, history.history['val_accuracy']
                    [best_epoch], color='r', label='Best Epoch (Validation)')
    axes[0].scatter(best_train_epoch, history.history['accuracy']
                    [best_train_epoch], color='g', label='Best Epoch (Train)')

    # Annotate the best validation accuracy point
    best_accuracy = history.history['val_accuracy'][best_epoch]
    random_offset_x = 10
    random_offset_y = np.random.randint(-30, 30)
    axes[0].annotate(f'{best_accuracy:.3f}',
                     xy=(best_epoch, best_accuracy),
                     xytext=(random_offset_x, random_offset_y),
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->'))

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].legend()

    # Annotate the best training accuracy point
    best_train_accuracy = history.history['accuracy'][best_train_epoch]
    random_offset_y = np.random.randint(-30, 30)
    axes[0].annotate(f'{best_train_accuracy:.3f}',
                     xy=(best_train_epoch, best_train_accuracy),
                     xytext=(random_offset_x, random_offset_y),
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->'))

    # Plot training and validation loss
    axes[1].plot(history.epoch, history.history['loss'], label='Train Loss')
    axes[1].plot(history.epoch, history.history['val_loss'],
                 label='Validation Loss')
    axes[1].scatter(best_epoch, history.history['val_loss']
                    [best_epoch], color='r', label='Best Epoch (Validation)')
    axes[1].scatter(best_train_epoch, history.history['loss']
                    [best_train_epoch], color='g', label='Best Epoch (Train)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].legend()

    # Annotate the best validation loss point
    best_loss = history.history['val_loss'][best_epoch]
    random_offset_y = np.random.randint(-30, 30)
    axes[1].annotate(f'{best_loss:.3f}',
                     xy=(best_epoch, best_loss),
                     xytext=(random_offset_x, random_offset_y),
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->'))

    # Annotate the best training loss point
    best_train_loss = history.history['loss'][best_train_epoch]
    random_offset_y = np.random.randint(-30, 30)
    axes[1].annotate(f'{best_train_loss:.3f}',
                     xy=(best_train_epoch, best_train_loss),
                     xytext=(random_offset_x, random_offset_y),
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->'))

    # Calculate best validation accuracy in percentage
    best_accuracy_percentage = round(best_accuracy * 100, 2)

    # String
    # Performance of accuracy and Training Time
    minutes = int(training_time // 60)
    seconds = int(training_time % 60)

    # model summary
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)

    # Set the text for the parameters on the right side
    performance = r'$\bf{' + f'Ref. name: {ref_name}' + '}$' + f"\nBest Validation Accuracy: {best_accuracy_percentage}%" + \
        f"\nBest Train Accuracy: {history.history['accuracy'][best_train_epoch] * 100:.2f}%" + \
        f", Training Time: {minutes} minutes {seconds} seconds"
    fig.suptitle(r'$\bf{' + title + '}$' + '\n' + performance, fontsize=18)

    fig.text(1.05, 1.00, r'$\bf{' + 'Config Parameters:' + '}$', fontsize=16,
             color='black', ha='left', transform=plt.gcf().transFigure)

    y_coord = 0.98  # Initial y-coordinate

    # Set the line height between each parameter group and each parameter within a group
    line_height = 0.028

    for group, params in config_param.items():
        fig.text(1.05, y_coord, r'$\bf{' + f'{group}:' + '}$', fontsize=14,
                 color='black', ha='left', transform=plt.gcf().transFigure)
        y_coord -= line_height

        for key, value in params.items():
            fig.text(1.05, y_coord, f"{key}: {value}", fontsize=12,
                     color='black', ha='left', transform=plt.gcf().transFigure)
            y_coord -= line_height

    fig.text(1.05, 0.01, r'$\bf{' + f'Model Summary:' + '}$' + f'\n{short_model_summary}',
             fontsize=10, color='black', ha='left', transform=plt.gcf().transFigure)

    plt.tight_layout()
    plt.show()
