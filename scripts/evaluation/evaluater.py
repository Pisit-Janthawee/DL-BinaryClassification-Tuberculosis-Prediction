
import numpy as np
import pandas as pd
import yaml
import json
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score, roc_curve, roc_auc_score, f1_score


class Evaluator:
    def __init__(self, estimator_name, ref_name, config_param, history, training_time):
        self.estimator_name = estimator_name
        self.ref_name = ref_name
        self.config_param = config_param
        self.history = history
        self.training_time = training_time

    def evaluate(self, estimator, X, y_actual, subset_name, threshold):

        # Model Prediction/Y hat
        y_pred = estimator.predict(
            X) if estimator else self.estimator.predict(X)
        # Handling Output Layer of Activation='sigmoid'
        threshold = threshold if threshold else 0.5
        y_pred_binary = (y_pred > threshold).astype(int).flatten()

        # Calculate ROC curve and AUC score
        fpr, tpr, thresholds = roc_curve(y_actual, y_pred_binary)
        auc_score = roc_auc_score(y_actual, y_pred_binary)
        # Classification Report
        report = classification_report(
            y_actual, y_pred_binary, output_dict=True)
        report_dict = json.loads(json.dumps(report))

        # Extract precision and recall from report_dict
        precision = report_dict['weighted avg']['precision']
        recall = report_dict['weighted avg']['recall']
        accuracy = report_dict['accuracy']
        f1 = report_dict['weighted avg']['f1-score']

        # Confusion Matrix
        cm = confusion_matrix(y_actual, y_pred_binary)
        TP = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TN = cm[1][1]
        total_seconds = self.training_time
        minutes = int(self.training_time // 60)
        seconds = int(self.training_time % 60)
        training_time_ = "{} minutes {} seconds".format(minutes, seconds)

        optimizer_params = self.config_param.get(
            'Optimizer parameters Configuration', {})
        hyperparams = self.config_param.get(
            'Hyperparameters Configuration', {})
        model_arch = self.config_param.get(
            'Model Architecture Configuration', {})
        model_training = self.config_param.get(
            'Model Training Configuration', {})

        accuracy_train = self.history.history['accuracy']
        accuracy_val = self.history.history['val_accuracy']

        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Dict
        performance_info = {
            'Model': self.estimator_name,
            'Timestamp': current_datetime,
            'Reference name': self.ref_name,
            'Optimizer parameters Configuration': optimizer_params,
            'Hyperparameters Configuration': hyperparams,
            'Model Architecture Configuration': model_arch,
            'Model Training Configuration': model_training,
            'accuracy_train': accuracy_train,
            'accuracy_val': accuracy_val,
            'Subset': subset_name,
            'Training time': training_time_,
            'Training in seconds': total_seconds,
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc_score,
            'Accuracy': accuracy,
        }

        return pd.DataFrame([performance_info]), y_pred

    def plot_confusion_matrix(self, y_actual, y_pred, subset_name):
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        cm = confusion_matrix(y_actual, y_pred_binary)
        labels = ['Negative', 'Positive']

        plt.figure(figsize=(18, 9))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {subset_name}')
        plt.show()

    def plot_roc(self, y_actual, y_pred, subset_name):
        fpr, tpr, thresholds = roc_curve(y_actual, y_pred)
        # Print ROC Curve
        plt.figure(figsize=(18, 9))
        plt.plot(fpr, tpr, marker='o', linestyle='-',)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {subset_name}')
        plt.grid()
        plt.show()
