import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Dict, Tuple
import pickle

from AbstractModel import Model

import seaborn as sns
sns.set_theme()

class ModelEvaluator:
    """
    A comprehensive model evaluation utility specifically designed to work with models
    that implement the Model abstract base class. This evaluator calculates various
    classification metrics and creates visualizations to help understand model performance.
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

        # Initialize metrics dictionary
        self.metrics = {}

        # Set plotting style

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate classification metrics including TPR, FPR, Precision, and F1 score
        using the model's predictions.

        Returns:
            Dictionary containing calculated metrics
        """
        # Calculate confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()

        # Calculate metrics with error handling for division by zero
        try:
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 1-Specificity
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision

            # Calculate F1 score with handling for edge cases
            if precision + tpr > 0:
                f1 = 2 * (precision * tpr) / (precision + tpr)
            else:
                f1 = 0

            accuracy = (tp + tn) / (tp + tn + fp + fn)

        except ZeroDivisionError:
            print("Warning: Division by zero encountered in metric calculation")
            return {}

        # Store and return metrics
        self.metrics = {
            'True Positive Rate (Sensitivity/Recall)': tpr,
            'False Positive Rate (1-Specificity)': fpr,
            'Precision': precision,
            'F1 Score': f1,
            'Accuracy': accuracy,
            'True Positives': tp,
            'False Positives': fp,
            'True Negatives': tn,
            'False Negatives': fn
        }

        return self.metrics

    def plot_confusion_matrix(self, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Create a visually appealing confusion matrix heatmap.

        Args:
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)

        # Calculate confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)

        # Create heatmap with improved styling
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )

        plt.title('Confusion Matrix', pad=20)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def plot_metric_comparison(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Create a bar plot comparing different metrics.

        Args:
            figsize: Figure size (width, height)
        """
        # Ensure metrics have been calculated
        if not self.metrics:
            self.calculate_metrics()

        plt.figure(figsize=figsize)

        # Select main metrics for comparison
        metrics_to_plot = {
            'TPR': self.metrics['True Positive Rate (Sensitivity/Recall)'],
            'Precision': self.metrics['Precision'],
            'F1 Score': self.metrics['F1 Score'],
            'Accuracy': self.metrics['Accuracy']
        }

        # Create bar plot with custom colors
        plt.bar(
            metrics_to_plot.keys(),
            metrics_to_plot.values(),
            color=['#2ecc71', '#e74c3c', '#f1c40f', '#3498db']
        )

        plt.title('Model Performance Metrics Comparison')
        plt.ylabel('Score')
        plt.ylim([0, 1])

        # Add value labels on top of each bar
        for i, v in enumerate(metrics_to_plot.values()):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

        plt.grid(True, axis='y')
        plt.show()

    def generate_evaluation_report(self) -> None:
        """
        Generate a comprehensive evaluation report including metrics and visualizations.
        This method serves as a one-stop solution for model evaluation.
        """
        # Calculate metrics
        metrics = self.calculate_metrics()

        # Print metrics report
        print("\nClassification Metrics Report")
        print("=" * 50)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")

        # Generate visualizations
        print("\nGenerating visualizations...")

        # Confusion Matrix
        print("\nConfusion Matrix:")
        self.plot_confusion_matrix()

        # Metrics Comparison
        print("\nMetrics Comparison:")
        self.plot_metric_comparison()


# Example usage:
if __name__ == "__main__":
    # Example implementation of the Model abstract class
    class ExampleModel(Model):
        def train(self, X, y, prev_model=None):
            # Simple majority class predictor for demonstration
            self.majority_class = np.bincount(y).argmax()

        def predict(self, X):
            return np.full(X.shape[0], self.majority_class)

        def saveModel(self, path_to_save):
            with open(path_to_save, 'wb') as f:
                pickle.dump(self, f)

        @classmethod
        def loadModel(cls, path_to_load):
            with open(path_to_load, 'rb') as f:
                return pickle.load(f)


    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Create and train model
    model = ExampleModel()
    model.train(X, y)

    # Create evaluator and generate report
    evaluator = ModelEvaluator(model, X, y)
    evaluator.generate_evaluation_report()