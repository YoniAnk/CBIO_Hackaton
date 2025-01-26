import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
from typing import Optional, Union
from Models.AbstractModel import Model


class KNNClassifier(Model):
    """
    A K-Nearest Neighbors classifier implementation using scikit-learn's NearestNeighbors as the backend.
    This implementation supports incremental learning through the prev_model parameter.
    """

    def __init__(self, n_neighbors: int = 5, metric: str = 'minkowski', p: int = 2):
        """
        Initialize the KNN classifier.

        Args:
            n_neighbors: Number of neighbors to use for classification
            metric: Distance metric to use (default is minkowski which gives Euclidean distance when p=2)
            p: Power parameter for minkowski metric
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.nn_model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=metric,
            p=p
        )
        self.X_train = None
        self.y_train = None

    def train_model(self, X: np.ndarray, y: np.ndarray, prev_model: Optional['KNNClassifier'] = None) -> None:
        """
        Train the KNN classifier. If prev_model is provided, combines the new data with the previous model's data.

        Args:
            X: Training data features of shape (n_samples, n_features)
            y: Training data labels of shape (n_samples,)
            prev_model: Optional previous model to continue training from
        """
        # Input validation
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays")

        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match")

        if prev_model is not None:
            # Combine with previous model's data
            self.X_train = np.vstack([prev_model.X_train, X])
            self.y_train = np.concatenate([prev_model.y_train, y])
        else:
            self.X_train = X
            self.y_train = y

        # Fit the nearest neighbors model
        self.nn_model.fit(self.X_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for test data X.

        Args:
            X: Test data features of shape (n_samples, n_features)

        Returns:
            Predicted labels of shape (n_samples,)
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Model must be trained before making predictions")

        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")

        # Find k nearest neighbors
        distances, indices = self.nn_model.kneighbors(X)

        # Get the labels of the nearest neighbors
        neighbor_labels = self.y_train[indices]

        # Perform majority voting
        predictions = np.array([
            np.bincount(neighbor_labels[i]).argmax()
            for i in range(len(X))
        ])

        return predictions

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
    def loadModel(path_to_load: str) -> 'KNNClassifier':
        """
        Load a saved model from a pickle file.

        Args:
            path_to_load: Path to the saved model file

        Returns:
            Loaded KNNClassifier model
        """
        if not path_to_load.endswith('.pkl'):
            path_to_load += '.pkl'

        with open(path_to_load, 'rb') as f:
            return pickle.load(f)


# Example usage:
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.rand(100, 2)  # 100 samples with 2 features
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Simple decision boundary

    # Create and train the model
    knn = KNNClassifier(n_neighbors=3)
    knn.train_model(X, y)

    # Make predictions
    X_test = np.random.rand(10, 2)
    predictions = knn.predict(X_test)
    print("Predictions:", predictions)

    # Save the model
    knn.saveModel("knn_model.pkl")

    # Load the model
    loaded_knn = KNNClassifier.loadModel("../../Desktop/CBIO HACKATHON/knn_model.pkl")

    # Continue training with new data
    X_new = np.random.rand(50, 2)
    y_new = (X_new[:, 0] + X_new[:, 1] > 1).astype(int)
    loaded_knn.train_model(X_new, y_new, prev_model=knn)