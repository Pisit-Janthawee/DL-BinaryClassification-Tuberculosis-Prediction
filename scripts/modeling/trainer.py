import pandas as pd
import numpy as np
import time
import os
import yaml
import re
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from scripts.experimentation.experiment_generator import ExperimentGenerator
from IPython.display import display, HTML


class Trainer:
    def __init__(self,
                 estimator,
                 estimator_name,
                 ref_name,
                 input_shape,
                 epochs,
                 batch_size,
                 estimator_config,
                 verbose=1,
                 metrics=['accuracy'],
                 loss='binary_crossentropy'):
        # Constructor to initialize Trainer class
        self.estimator = estimator
        self.estimator_name = estimator_name
        self.estimator_config = estimator_config
        self.ref_name = ref_name
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.metrics = metrics
        self.loss = loss
        self.optimizer = None
        self.optimizer_config = None
        self.optimizer_name = None
        self.training_time = None

    def fit(self, X_train, y_train, X_val, y_val,experiment_save=True, verbose=1):
        # Method to fit the model to the data
        start_time = time.time()
       
        if verbose == 1:
            config_df = pd.DataFrame([self.get_config()])
            display(HTML(config_df.to_html()))

        # Compile the model
        self.estimator.compile(loss=self.loss,
                               optimizer=self.optimizer,
                               metrics=self.metrics)

        # Train the model
        history = self.estimator.fit(X_train, y_train, epochs=self.epochs,
                                     batch_size=self.batch_size,
                                     validation_data=(X_val, y_val),
                                     verbose=verbose)

        end_time = time.time()
        self.training_time = end_time - start_time
        self.history = history
        if experiment_save == True:
            # Create an ExperimentGenerator object
            experimenter = ExperimentGenerator(estimator_name=self.estimator_name,
                                ref_name=self.ref_name,
                                estimator=self.estimator,
                                get_config=self.get_config(),
                                history=history,
                                training_time=self.training_time)
            experimenter.evaluate_and_save_experiment(X_train=X_train, 
                                                    y_train=y_train, 
                                                    X_val=X_val, 
                                                    y_val=y_val)
        return history

    def get_config(self):
        # Method to get the configuration parameters of the model
        optimizer_config = self.optimizer.get_config()

        # Configuration dictionary
        config_param = {
            'Optimizer parameters Configuration': {
                'optimizer_name': self.optimizer_name,
                'learning_rate': optimizer_config.get('learning_rate', 0.001),
                'beta1': optimizer_config.get('beta_1', 0.9),
                'beta2': optimizer_config.get('beta_2', 0.999),
                'epsilon': optimizer_config.get('epsilon', 1e-08),
                'momentum': optimizer_config.get('momentum', None),
                'rho': optimizer_config.get('rho', None),
            },
            'Hyperparameters Configuration': {
                'conv_padding': self.estimator_config['conv_padding'],
                'conv_kernel_size': self.estimator_config['conv_kernel_size'],
                'conv_stride': self.estimator_config['conv_stride'],
                'pool_padding': self.estimator_config['pool_padding'],
                'pool_kernel_size': self.estimator_config['pool_kernel_size'],
                'pool_stride': self.estimator_config['pool_stride'],
                'pooling_type': self.estimator_config['pooling_type'],
            },
            'Model Architecture Configuration': {
                'input_shape': self.estimator_config['input_shape'],
                'unit_size_rate': self.estimator_config['unit_size_rate']
            },
            'Model Training Configuration': {
                'batch_size': self.batch_size,
                'epoch': self.epochs
            },
        }
        return config_param

    def save_artifact(self, model_name):
        # Method to save the trained model as an artifact
        formatted_model_name = re.sub(r'[^\w\s]', '', model_name)
        formatted_model_name = formatted_model_name.replace(' ', '_').lower()
        os.makedirs('artifacts', exist_ok=True)
        save_path = os.path.abspath(f'artifacts/{formatted_model_name}.h5')
        self.estimator.save(save_path)
        print(f"Model: {formatted_model_name}\n-> saved at '{save_path}'")

    def choose_optimizer(self,
                         opt_name='adam',
                         learning_rate=None,
                         beta1=None,
                         beta2=None,
                         epsilon=None,
                         momentum=None,
                         rho=None,):
        # Method to choose an optimizer for the model
        if opt_name.lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate if learning_rate else 0.001,
                beta_1=beta1 if beta1 else 0.9,
                beta_2=beta2 if beta2 else 0.999,
                epsilon=epsilon if epsilon else 1e-08,
            )
        elif opt_name.lower() == 'sgd':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate if learning_rate else 0.01,
                momentum=momentum if momentum else 0.9
            )
        elif opt_name.lower() == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=learning_rate if learning_rate else 0.001,
                rho=rho if rho else 0.9,
                momentum=momentum if momentum else 0.0,
                epsilon=epsilon if epsilon else 1e-08
            )
        elif opt_name.lower() == 'adadelta':
            optimizer = tf.keras.optimizers.Adadelta(
                learning_rate=learning_rate if learning_rate else 1.0,
                rho=rho if rho else 0.95,
                epsilon=epsilon if epsilon else 1e-08
            )
        elif opt_name.lower() == 'adagrad':
            optimizer = tf.keras.optimizers.Adagrad(
                learning_rate=learning_rate if learning_rate else 0.01,
                initial_accumulator_value=0.1
            )
        elif opt_name.lower() == 'adamax':
            optimizer = tf.keras.optimizers.Adamax(
                learning_rate=learning_rate if learning_rate else 0.002,
                beta_1=beta1 if beta1 else 0.9,
                beta_2=beta2 if beta2 else 0.999,
                epsilon=epsilon if epsilon else 1e-08
            )
        elif opt_name.lower() == 'nadam':
            optimizer = tf.keras.optimizers.Nadam(
                learning_rate=learning_rate if learning_rate else 0.002,
                beta_1=beta1 if beta1 else 0.9,
                beta_2=beta2 if beta2 else 0.999,
                epsilon=epsilon if epsilon else 1e-08
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")
        self.optimizer = optimizer
        self.optimizer_name = opt_name

    def get_history(self, title):
        '''
        Method to visualize training history.

        :Parameters:
            - self.history (object): History object returned by the model.fit() method.
            - self.estimator_name (String): name of model
            - self.ref_name (String): reference name
            - self.get_config() (Dict): Hyperparameters configuration and optimizer parameters.
            - self.training_time (float): Time taken for training in seconds.
        '''
        # Get the index of the epoch with the highest validation accuracy
        best_epoch = np.argmax(self.history.history['val_accuracy'])
        best_train_epoch = np.argmax(self.history.history['accuracy'])

        # Create a 1x2 grid of subplots
        fig, axes = plt.subplots(1, 2, figsize=(24, 15))

        # Plot training and validation accuracy
        axes[0].plot(self.history.epoch, self.history.history['accuracy'],
                     label='Train Accuracy')
        axes[0].plot(self.history.epoch, self.history.history['val_accuracy'],
                     label='Validation Accuracy')
        axes[0].scatter(best_epoch, self.history.history['val_accuracy']
                        [best_epoch], color='r', label='Best Epoch (Validation)')
        axes[0].scatter(best_train_epoch, self.history.history['accuracy']
                        [best_train_epoch], color='g', label='Best Epoch (Train)')

        # Annotate the best validation accuracy point
        best_accuracy = self.history.history['val_accuracy'][best_epoch]
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
        best_train_accuracy = self.history.history['accuracy'][best_train_epoch]
        random_offset_y = np.random.randint(-30, 30)
        axes[0].annotate(f'{best_train_accuracy:.3f}',
                         xy=(best_train_epoch, best_train_accuracy),
                         xytext=(random_offset_x, random_offset_y),
                         textcoords='offset points',
                         arrowprops=dict(arrowstyle='->'))

        # Plot training and validation loss
        axes[1].plot(self.history.epoch,
                     self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.epoch, self.history.history['val_loss'],
                     label='Validation Loss')
        axes[1].scatter(best_epoch, self.history.history['val_loss']
                        [best_epoch], color='r', label='Best Epoch (Validation)')
        axes[1].scatter(best_train_epoch, self.history.history['loss']
                        [best_train_epoch], color='g', label='Best Epoch (Train)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training and Validation Loss')
        axes[1].legend()

        # Annotate the best validation loss point
        best_loss = self.history.history['val_loss'][best_epoch]
        random_offset_y = np.random.randint(-30, 30)
        axes[1].annotate(f'{best_loss:.3f}',
                         xy=(best_epoch, best_loss),
                         xytext=(random_offset_x, random_offset_y),
                         textcoords='offset points',
                         arrowprops=dict(arrowstyle='->'))

        # Annotate the best training loss point
        best_train_loss = self.history.history['loss'][best_train_epoch]
        random_offset_y = np.random.randint(-30, 30)
        axes[1].annotate(f'{best_train_loss:.3f}',
                         xy=(best_train_epoch, best_train_loss),
                         xytext=(random_offset_x, random_offset_y),
                         textcoords='offset points',
                         arrowprops=dict(arrowstyle='->'))

        # Calculate best validation accuracy in percentage
        best_accuracy_percentage = round(best_accuracy * 100, 2)

        # String for performance, accuracy, and training time
        minutes = int(self.training_time // 60)
        seconds = int(self.training_time % 60)

        # model summary
        stringlist = []
        self.estimator.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)

        # Set the text for the parameters on the right side
        performance = r'$\bf{' + f'Ref. name: {self.ref_name}' + '}$' + f"\nBest Validation Accuracy: {best_accuracy_percentage}%" + \
            f"\nBest Train Accuracy: {self.history.history['accuracy'][best_train_epoch] * 100:.2f}%" + \
            f", Training Time: {minutes} minutes {seconds} seconds"
        fig.suptitle(r'$\bf{' + title + '}$' +
                     '\n' + performance, fontsize=18)

        fig.text(1.05, 1.00, r'$\bf{' + 'Config Parameters:' + '}$', fontsize=16,
                 color='black', ha='left', transform=plt.gcf().transFigure)

        y_coord = 0.98  # Initial y-coordinate

        # Set the line height between each parameter group and each parameter within a group
        line_height = 0.028

        for group, params in self.get_config().items():
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
