import pandas as pd
import yaml
import re
import os
from IPython.display import display, HTML

from scripts.evaluation.evaluater import Evaluator


class ExperimentGenerator:
    def __init__(self, estimator_name, ref_name, estimator, get_config, history, training_time):
        self.estimator_name = estimator_name
        self.ref_name = ref_name
        self.estimator = estimator
        self.get_config = get_config
        self.history = history
        self.training_time = training_time

    def get_exp_index(self):
        experiment_folder = 'repositorys/metadata/experiment'

        files = os.listdir(experiment_folder)

        model_files = [f for f in files if f.startswith(f"experiment")]

        existing_indices = [int(re.search(r'\d+', f).group())
                            for f in model_files]
        next_index = max(existing_indices) + 1 if existing_indices else 1

        return next_index


    def evaluate_and_save_experiment(self, X_train, y_train, X_val, y_val):
        evaluator = Evaluator(estimator_name=self.estimator_name,
                              ref_name=self.ref_name,
                              config_param=self.get_config,
                              history=self.history,
                              training_time=self.training_time)

        exp_index = self.get_exp_index()
        experiment_name = f"experiment{exp_index}"

        train_performance_df, y_pred_train = evaluator.evaluate(
            estimator=self.estimator,
            X=X_train, y_actual=y_train, subset_name='Train', threshold=None)

        val_performance_df, y_pred_test = evaluator.evaluate(estimator=self.estimator,
                                                X=X_val, y_actual=y_val, subset_name='Validation', threshold=None)
        combined_performance_df = pd.concat(
            [train_performance_df, val_performance_df], ignore_index=True)
        combined_performance_df.insert(0, 'experiment', exp_index)

        with open(f'repositorys/metadata/experiment/{experiment_name}.yaml', 'w') as file:
            yaml.dump(
                combined_performance_df.to_dict(orient='records'),
                file,
                sort_keys=False, width=72, indent=4,
                default_flow_style=None)

