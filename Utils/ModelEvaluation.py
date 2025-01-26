import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, Tuple
from Models.AbstractModel import Model
import seaborn as sns

sns.set_theme()


class ModelEvaluator:
    """
    An enhanced model evaluation utility designed to work with multi-class classification models
    that implement the Model abstract base class. This evaluator calculates various classification
    metrics and creates visualizations to help understand model performance across multiple classes.
    """

    def __init__(self, model: Model, X: np.ndarray, y: np.ndarray):
        """
        Initialize the evaluator with a model and data.

        Args:
            model: An instance of a class implementing the Model abstract base class
            X: Feature data for evaluation
            y: True labels for evaluation
        """
        # if not isinstance(model, Model):
        #     raise TypeError("Model must be an instance of the Model abstract base class")

        self.model = model
        self.X = X
        self.y_true = y
        self.y_pred = model.predict(X)

        # Get unique classes
        self.classes = np.unique(np.concatenate([self.y_true, self.y_pred]))
        self.n_classes = len(self.classes)

        # Initialize metrics dictionary
        self.metrics = {}

    def calculate_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate classification metrics for each class including precision, recall, and F1 score.
        Returns:
            Dictionary containing calculated metrics for each class
        """
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)

        # Initialize metrics dictionary for each class
        class_metrics = {}

        for i, class_label in enumerate(self.classes):
            # Calculate true positives, false positives, and false negatives for current class
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - (tp + fp + fn)

            # Calculate metrics with error handling
            try:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (tp + tn) / np.sum(cm)

                class_metrics[f'Class {class_label}'] = {
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1,
                    'Accuracy': accuracy,
                    'True Positives': tp,
                    'False Positives': fp,
                    'True Negatives': tn,
                    'False Negatives': fn
                }

            except ZeroDivisionError:
                print(f"Warning: Division by zero encountered in metric calculation for class {class_label}")
                continue

        return class_metrics

    def plot_confusion_matrix(self, model_name:str = "",  figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Create a confusion matrix heatmap showing both normalized and original values.

        Args:
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)

        # Calculate confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)

        # Calculate normalized values
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create annotations with both values
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm_norm[i, j]:.2f}\n({cm[i, j]})'

        # Create heatmap using normalized values for colors
        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt='',
            cmap='Blues',
            xticklabels=[f'Class {c}' for c in self.classes],
            yticklabels=[f'Class {c}' for c in self.classes],
            vmin=0,
            vmax=1
        )

        plt.title(f'{model_name} Confusion Matrix', pad=20)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def plot_class_metrics(self, figsize: Tuple[int, int] = (8, 5)) -> None:
        class_metrics = self.calculate_class_metrics()
        metrics_to_plot = ['Precision', 'Recall', 'F1 Score']

        values_by_metric = {metric: [class_metrics[f'Class {c}'][metric]
                                     for c in self.classes]
                            for metric in metrics_to_plot}

        min_val = min(min(vals) for vals in values_by_metric.values())
        max_val = max(max(vals) for vals in values_by_metric.values())
        y_margin = (max_val - min_val) * 0.1

        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(self.classes))
        width = 0.25

        for i, (metric, values) in enumerate(values_by_metric.items()):
            ax.bar(x + i * width, values, width, label=metric, alpha=0.8)

        ax.set_ylim(max(0, min_val - y_margin), min(1, max_val + y_margin))
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'Class {c}' for c in self.classes])

        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def generate_evaluation_report(self, normalize_cm: bool = False,
                                   model_name: str = "") -> None:
        """
        Generate a comprehensive evaluation report including per-class metrics and visualizations.

        Args:
            normalize_cm: If True, display normalized confusion matrix
        """
        # Calculate per-class metrics
        class_metrics = self.calculate_class_metrics()

        # Print detailed report
        print("\nClassification Metrics Report")
        print("=" * 50)

        for class_label, metrics in class_metrics.items():
            print(f"\n{class_label} Metrics:")
            print("-" * 30)
            for metric, value in metrics.items():
                if isinstance(value, (int, np.integer)):
                    print(f"{metric}: {value}")
                else:
                    print(f"{metric}: {value:.3f}")

        # Generate visualizations
        print("\nGenerating visualizations...")

        # Confusion Matrix
        print("\nConfusion Matrix:")
        self.plot_confusion_matrix(model_name=model_name)

        # Class Metrics Comparison
        print("\nClass Metrics Comparison:")
        self.plot_class_metrics()

        # Print sklearn classification report for additional metrics
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_true, self.y_pred))