import numpy as np
from scipy.special import softmax
import pickle
from typing import Optional, Tuple
from Models.AbstractModel import Model


class LogisticRegression(Model):
    """
    A Logistic Regression classifier implementation supporting both binary and multiclass classification.
    Uses gradient descent optimization with optional L2 regularization.

    The model implements the following key features:
    - Binary and multiclass classification using one-vs-all approach
    - L2 regularization to prevent overfitting
    - Mini-batch gradient descent optimization
    - Support for continuous training with previous model parameters
    """

    def __init__(
            self,
            learning_rate: float = 0.01,
            n_iterations: int = 1000,
            batch_size: int = 32,
            lambda_reg: float = 0.01
    ):
        """
        Initialize the Logistic Regression model.

        Args:
            learning_rate: Step size for gradient descent optimization
            n_iterations: Number of training iterations
            batch_size: Size of mini-batches for gradient descent
            lambda_reg: L2 regularization strength
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg

        # These will be initialized during training
        self.weights = None
        self.bias = None
        self.n_classes = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid function for binary classification.

        Args:
            z: Input values

        Returns:
            Sigmoid activation values
        """
        # Clip values to avoid numerical overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _initialize_parameters(self, n_features: int) -> None:
        """
        Initialize model parameters (weights and bias).

        Args:
            n_features: Number of input features
        """
        if self.n_classes == 2:
            # Binary classification: single set of weights
            self.weights = np.zeros((n_features, 1))
            self.bias = np.zeros(1)
        else:
            # Multiclass: one set of weights per class
            self.weights = np.zeros((n_features, self.n_classes))
            self.bias = np.zeros(self.n_classes)

    def _compute_gradients(
            self,
            X_batch: np.ndarray,
            y_batch: np.ndarray,
            y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients for mini-batch gradient descent.

        Args:
            X_batch: Input features for current batch
            y_batch: True labels for current batch
            y_pred: Predicted probabilities for current batch

        Returns:
            Tuple of (weight gradients, bias gradients)
        """
        m = X_batch.shape[0]

        # For binary classification
        if self.n_classes == 2:
            error = y_pred - y_batch.reshape(-1, 1)
            dw = (1 / m) * np.dot(X_batch.T, error) + (self.lambda_reg * self.weights)
            db = (1 / m) * np.sum(error)
        # For multiclass classification
        else:
            # Convert y_batch to one-hot encoding
            y_one_hot = np.zeros((m, self.n_classes))
            y_one_hot[np.arange(m), y_batch] = 1

            error = y_pred - y_one_hot
            dw = (1 / m) * np.dot(X_batch.T, error) + (self.lambda_reg * self.weights)
            db = (1 / m) * np.sum(error, axis=0)

        return dw, db

    def train_model(
            self,
            X: np.ndarray,
            y: np.ndarray,
            prev_model: Optional['LogisticRegression'] = None
    ) -> None:
        """
        Train the logistic regression model using mini-batch gradient descent.

        Args:
            X: Training data features of shape (n_samples, n_features)
            y: Training data labels of shape (n_samples,)
            prev_model: Optional previous model to continue training from
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays")

        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match")

        # Determine number of classes
        self.n_classes = len(np.unique(y))

        # Initialize or load parameters
        if prev_model is not None:
            self.weights = prev_model.weights.copy()
            self.bias = prev_model.bias.copy()
        elif self.weights is None:
            self._initialize_parameters(X.shape[1])

        # Training loop with mini-batch gradient descent
        n_samples = X.shape[0]

        for _ in range(self.n_iterations):
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch processing
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                # Forward pass
                if self.n_classes == 2:
                    z = np.dot(X_batch, self.weights) + self.bias
                    y_pred = self._sigmoid(z)
                else:
                    z = np.dot(X_batch, self.weights) + self.bias
                    y_pred = softmax(z, axis=1)

                # Compute gradients
                dw, db = self._compute_gradients(X_batch, y_batch, y_pred)

                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X: Test data features of shape (n_samples, n_features)

        Returns:
            Predicted labels of shape (n_samples,)
        """
        if self.weights is None:
            raise RuntimeError("Model must be trained before making predictions")

        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")

        # Binary classification
        if self.n_classes == 2:
            z = np.dot(X, self.weights) + self.bias
            probabilities = self._sigmoid(z)
            return (probabilities >= 0.5).astype(int).flatten()

        # Multiclass classification
        else:
            z = np.dot(X, self.weights) + self.bias
            probabilities = softmax(z, axis=1)
            return np.argmax(probabilities, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Args:
            X: Test data features of shape (n_samples, n_features)

        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        if self.weights is None:
            raise RuntimeError("Model must be trained before making predictions")

        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")

        # Binary classification
        if self.n_classes == 2:
            z = np.dot(X, self.weights) + self.bias
            prob_class_1 = self._sigmoid(z)
            return np.hstack([1 - prob_class_1, prob_class_1])

        # Multiclass classification
        else:
            z = np.dot(X, self.weights) + self.bias
            return softmax(z, axis=1)

    def saveModel(self, path_to_save: str) -> None:
        """
        Save the model to a pickle file.

        Args:
            path_to_save: Path where the model should be saved
        """
        if not path_to_save.endswith('.pkl'):
            path_to_save += '.pkl'

        with open(path_to_save, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def loadModel(path_to_load: str) -> 'LogisticRegression':
        """
        Load a saved model from a pickle file.

        Args:
            path_to_load: Path to the saved model file

        Returns:
            Loaded LogisticRegression model
        """
        if not path_to_load.endswith('.pkl'):
            path_to_load += '.pkl'

        with open(path_to_load, 'rb') as f:
            return pickle.load(f)


# Example usage:
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 5)  # 1000 samples with 5 features

    # Binary classification example
    y_binary = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Create and train binary classifier
    model_binary = LogisticRegression(
        learning_rate=0.01,
        n_iterations=1000,
        batch_size=32,
        lambda_reg=0.01
    )
    model_binary.train_model(X, y_binary)

    # Make predictions
    predictions = model_binary.predict(X[:10])
    probabilities = model_binary.predict_proba(X[:10])

    print("Binary Classification Results:")
    print("Predictions:", predictions)
    print("Probabilities:", probabilities)

    # Multiclass classification example
    y_multi = (X[:, 0] + X[:, 1] > 1).astype(int) + (X[:, 2] + X[:, 3] > 1).astype(int)

    # Create and train multiclass classifier
    model_multi = LogisticRegression(
        learning_rate=0.01,
        n_iterations=1000,
        batch_size=32,
        lambda_reg=0.01
    )
    model_multi.train_model(X, y_multi)

    # Make predictions
    predictions_multi = model_multi.predict(X[:10])
    probabilities_multi = model_multi.predict_proba(X[:10])

    print("\nMulticlass Classification Results:")
    print("Predictions:", predictions_multi)
    print("Probabilities:", probabilities_multi)