# Utility
from scripts.modeling.trainer import Trainer


# CNNs
import tensorflow as tf
import keras

# Common
import time
import datetime
import pandas as pd
import numpy as np
import random
import itertools
from IPython.display import display, HTML

# Model Training
from sklearn.model_selection import KFold

# Visual
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

class Tuner:
    def __init__(self, estimator, param_grid=None, input_shape=None, num_folds=5, epochs=10, batch_size=32):
        self.estimator = estimator
        self.param_grid = param_grid
        self.input_shape = input_shape
        self.num_folds = num_folds
        self.epochs = epochs
        self.batch_size = batch_size
        self.best_estimator_ = None

    def fit(self, X, y):
        best_accuracy = 0
        best_hyperparameters = None
        tuning_results = []

        total_combinations = np.prod([len(vals)
                                     for vals in self.param_grid.values()])
        pbar_combinations = tqdm(total=total_combinations, desc="Tuning...")

        combination_count = 0
        # Get current time
        start_time = time.time()

        for index, params in enumerate(self.param_grid_generator_()):
            pbar_combinations.update(1)
            if isinstance(params.get('input_shape'), tuple):
                            # Create a DataFrame with 'Parameter' and 'Value' columns
                params_df = pd.DataFrame(
                    {'Parameter': ['input_shape'], 'Value': [params['input_shape']]})
                display(HTML(params_df.to_html()))
            else:
                params_df = pd.DataFrame.from_dict(
                    params, orient='index', columns=['Value'])
                display(HTML(params_df.to_html()))

            cv_accuracies_train = []
            cv_accuracies_val = []
            cv_losses_train = []
            cv_losses_val = []
            cv_training_times = []

            kf = KFold(n_splits=self.num_folds, shuffle=True)
            for fold_idx, (train_indices, val_indices) in enumerate(kf.split(X)):
                process = f"K-fold ({fold_idx+1}/{self.num_folds})"
                pbar_kfold = tqdm(total=self.num_folds,
                                  desc=process, leave=False)
                pbar_kfold.update(1)
                try:
                    X_resized = tf.image.resize(X, params.get(
                        'input_shape', self.input_shape)).numpy()
                    X_train, X_val = X_resized[train_indices], X_resized[val_indices]
                    y_train, y_val = y[train_indices], y[val_indices]

                    model_config = {
                        'input_shape': params.get('input_shape', self.input_shape),
                        'unit_size_rate': params.get('unit_size_rate', 0.05),
                        'l1_lambda': params.get('l1_lambda', None),
                        'l2_lambda': params.get('l2_lambda', None),

                        'conv_padding': params.get('conv_padding', 'same'),
                        'conv_kernel_size': params.get('conv_kernel_size', (3, 3)),
                        'conv_stride': params.get('conv_stride', 1),

                        'pool_padding': params.get('pool_padding', 'same'),
                        'pool_kernel_size': params.get('pool_kernel_size', (2, 2)),
                        'pool_stride': params.get('pool_stride', 2),

                        'dropout':  params.get('dropout', 0),
                        'pooling_type': params.get('pooling_type', 'max')
                    }

                    # Create the CNN model based on the model_builder function
                    model = self.estimator(**model_config).build_model()

                    train_config = {
                        'estimator': model,
                        'estimator_name': 'experiment',
                        'ref_name': 'Original dataset',
                        'input_shape': params.get('input_shape', self.input_shape),
                        'epochs': params.get('epochs', self.epochs),
                        'batch_size': params.get('batch_size', self.batch_size),
                        'estimator_config': model_config,
                    }
                    trainer = Trainer(**train_config)
                    trainer.choose_optimizer(opt_name=params.get('opt_name', 'adam'),
                                             learning_rate=params.get(
                                                 'learning_rate', 0.001),
                                             beta1=params.get(
                        'beta1', None),
                        beta2=params.get('beta2', None),
                        epsilon=params.get(
                        'epsilon', None),
                        momentum=params.get(
                        'momentum', None),
                        rho=params.get(
                        'rho', None),
                    )
                    history = trainer.fit(X_train=X_train, 
                                            y_train=y_train,
                                            X_val=X_val, 
                                            y_val=y_val, 
                                            experiment_save=False,
                                            verbose=0)
                    
                    
                    accuracy_train = history.history['accuracy']
                    accuracy_val = history.history['val_accuracy']
                    loss_train = history.history['loss']
                    loss_val = history.history['val_loss'] 

                    cv_accuracies_train.append(accuracy_train)
                    cv_accuracies_val.append(accuracy_val)
                    cv_losses_train.append(loss_train)  
                    cv_losses_val.append(loss_val)  

                    cv_training_times.append(trainer.training_time)
                    pbar_combinations.close()
                except Exception as e:
                    print(f"Error occurred in Fold {fold_idx + 1}: {e}")
                    continue  
            combination_count += 1
            # Convert list to Numpy array
            avg_accuracies_train_array = np.array(cv_accuracies_train)
            avg_accuracies_val_array = np.array(cv_accuracies_val)
            avg_losses_train_array = np.array(cv_losses_train)
            avg_losses_val_array = np.array(cv_losses_val)
            # List of average
            avg_accuracies_train = np.mean(avg_accuracies_train_array, axis=0)
            avg_accuracies_val = np.mean(avg_accuracies_val_array, axis=0)
            avg_losses_train = np.mean(avg_losses_train_array, axis=0)
            avg_losses_val = np.mean(avg_losses_val_array, axis=0)
            # Single average
            avg_accuracy_train = np.mean(cv_accuracies_train)
            avg_accuracy_val = np.mean(cv_accuracies_val)
            training_time = np.mean(cv_training_times)

            # Time
            elapsed_time = time.time() - start_time
            avg_time_per_combination = elapsed_time / combination_count
            estimated_total_time = avg_time_per_combination * total_combinations
            finish_time_seconds = start_time + estimated_total_time
            finish_time_hours = estimated_total_time / 3600  # Convert to hours

            # Print estimated finish time for the entire tuning process
            print(
                f"Estimated Finish time in {finish_time_hours:.2f} hours / ~{finish_time_hours*60:.2f} minutes")
            # Print estimated finish time and other information
            print(
                f"Estimated Finish Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(finish_time_seconds))}")
            
            minutes = int(training_time // 60)  
            seconds = int(training_time % 60)
            training_time_ = "{} minutes {} seconds".format(minutes, seconds)

            optimizer_params = trainer.get_config().get(
                'Optimizer parameters Configuration', {})
            hyperparams = trainer.get_config().get('Hyperparameters Configuration', {})
            model_arch = trainer.get_config().get(
                'Model Architecture Configuration', {})
            model_training = trainer.get_config().get(
                'Model Training Configuration', {})

            # Dict
            
            performance_info = {
                'Model': f'{combination_count}',
                'Optimizer parameters Configuration': optimizer_params,
                'Hyperparameters Configuration': hyperparams,
                'Model Architecture Configuration': model_arch,
                'Model Training Configuration': model_training,
                'accuracy_train': avg_accuracies_train.tolist(),
                'accuracy_val': avg_accuracies_val.tolist(),
                'loss_train': avg_losses_train.tolist(), 
                'loss_val': avg_losses_val.tolist(),  
                'Training time': training_time_,
                'Training in seconds': training_time,
                'Accuracy_train': avg_accuracy_train,
                'Accuracy_val': avg_accuracy_val,
            }
            
            if avg_accuracy_val > best_accuracy:
                best_accuracy = avg_accuracy_val
                best_hyperparameters = params
                self.best_estimator_ = model
            tuning_results.append(performance_info)

        pbar_combinations.close()
        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'Finish Tunning!! at {current_datetime}')
        self.best_hyperparameters = best_hyperparameters
        self.best_accuracy = best_accuracy
        self.tuning_results = tuning_results

    def param_grid_generator_(self):
        keys = self.param_grid.keys()
        values_list = self.param_grid.values()
        for combination in itertools.product(*values_list):
            yield dict(zip(keys, combination))

    def best_estimator_(self):
        return self.best_estimator_
    
    def best_hyperparameters_(self):
        return self.best_hyperparameters

    def best_accuracy(self):
        return self.best_accuracy

    def tuning_results(self):
        return self.tuning_results

    def get_pandas(self):
        return pd.DataFrame(self.tuning_results)
    
    def plot_accuracy_comparison(self, title, configuration_key, hyperparameter_name, performance):
        fig, ax = plt.subplots(figsize=(18, 9))
        legend_labels = []  # Store legend labels

        for index, row in self.get_pandas().iterrows():
            # Get the accuracy list (train or val) based on 'performance' parameter
            if performance.lower() == "train":
                accuracy_list = row['accuracy_train']
            elif performance.lower() in ["val", "validation"]:
                accuracy_list = row['accuracy_val']
            else:
                raise ValueError("Invalid performance parameter. Use 'train' or 'val'.")

            # Extract the hyperparameter value from the specified configuration key
            if configuration_key and hyperparameter_name:
                configuration_dict = row.get(configuration_key, {})
                hyperparameter_value = configuration_dict.get(hyperparameter_name, "N/A")
                config = f"{hyperparameter_name}: {hyperparameter_value}"
            elif hyperparameter_name is None:
                configuration_dict = row.get(configuration_key, {})
                config = f"{configuration_dict}"

            else:
                config = ""

            # Plot the accuracy list
            line, = plt.plot(accuracy_list, label=f'Model {row["Model"]} ({max(accuracy_list):.3f})')
            legend_labels.append(f'Model {row["Model"]} {config}')

            # Annotate the highest accuracy point
            best_epoch = accuracy_list.index(max(accuracy_list))
            best_accuracy = max(accuracy_list)

            # Randomize xytext coordinates
            random_offset_x = random.randint(-30, 30)
            random_offset_y = random.randint(-30, 30)

            plt.annotate(f'{best_accuracy:.3f}',
                         xy=(best_epoch, best_accuracy),
                         # Randomize the offset
                         xytext=(random_offset_x, random_offset_y),
                         textcoords='offset points',
                         arrowprops=dict(arrowstyle='->'))

        # Add labels and legend
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')  # You can customize the ylabel here
        plt.title(title)
        plt.legend(legend_labels)
        plt.show()


