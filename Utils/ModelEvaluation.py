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
        if not isinstance(model, Model):
            raise TypeError("Model must be an instance of the Model abstract base class")

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
        print('#####################################')
        cm = confusion_matrix(self.y_true, self.y_pred)
        print(cm)
        print('#####################################')

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

    def plot_confusion_matrix(self, figsize: Tuple[int, int] = (10, 8),
                              normalize: bool = False) -> None:
        """
        Create a visually appealing confusion matrix heatmap for multiple classes.

        Args:
            figsize: Figure size (width, height)
            normalize: If True, normalize confusion matrix by row
        """
        plt.figure(figsize=figsize)

        # Calculate confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)

        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'

        # Create heatmap with improved styling
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=[f'Class {c}' for c in self.classes],
            yticklabels=[f'Class {c}' for c in self.classes]
        )

        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''), pad=20)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def plot_class_metrics(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Create a grouped bar plot comparing different metrics across classes.

        Args:
            figsize: Figure size (width, height)
        """
        # Calculate metrics for each class
        class_metrics = self.calculate_class_metrics()

        # Prepare data for plotting
        metrics_to_plot = ['Precision', 'Recall', 'F1 Score']
        x = np.arange(len(self.classes))
        width = 0.25

        plt.figure(figsize=figsize)

        # Plot bars for each metric
        for i, metric in enumerate(metrics_to_plot):
            values = [class_metrics[f'Class {c}'][metric] for c in self.classes]
            plt.bar(x + i * width, values, width, label=metric)

        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Performance Metrics by Class')
        plt.xticks(x + width, [f'Class {c}' for c in self.classes])
        plt.legend()
        plt.grid(True, axis='y')
        plt.show()

    def generate_evaluation_report(self, normalize_cm: bool = False) -> None:
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
        self.plot_confusion_matrix(normalize=normalize_cm)

        # Class Metrics Comparison
        print("\nClass Metrics Comparison:")
        self.plot_class_metrics()

        # Print sklearn classification report for additional metrics
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_true, self.y_pred))