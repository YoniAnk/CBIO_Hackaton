from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class Model(ABC):
    """
    Abstract base class for classification models. This class defines the required interface
    that all model implementations must follow. By inheriting from this class, we ensure
    consistent behavior across different model implementations.
    """

    @abstractmethod
    def train_model(self, X: np.ndarray, y: np.ndarray, prev_model: Optional['Model'] = None) -> None:
        """
        Train the model on the provided data.

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
            prev_model: Optional previous model to continue training from
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Features to predict on, shape (n_samples, n_features)

        Returns:
            Predicted labels of shape (n_samples,)
        """
        pass

    @abstractmethod
    def saveModel(self, path_to_save: str) -> None:
        """
        Save the model to a file.

        Args:
            path_to_save: Path where the model should be saved
        """
        pass

    @classmethod
    @abstractmethod
    def loadModel(cls, path_to_load: str) -> 'Model':
        """
        Load a model from a file.

        Args:
            path_to_load: Path to the saved model file

        Returns:
            Loaded model instance
        """
        pass