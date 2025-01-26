import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class MLModel:
    def __init__(self, task='classification'):
        """
        Initialize the model with configurable parameters
        
        Args:
        - task (str): 'classification' or 'regression'
        """
        # Global hyperparameters with default values
        self.params = {
            'n_estimators': 100,
            'max_depth': 5,
            'random_state': 42,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'auto'
        }
        
        self.task = task
        self.model = None
    
    def set_params(self, **kwargs):
        """
        Update model parameters
        
        Args:
        - **kwargs: Key-value pairs of parameters to update
        """
        self.params.update(kwargs)
    
    def train(self, X, y, prev_model=None):
        """
        Train the model
        
        Args:
        - X (np.array): Training features
        - y (np.array): Training labels
        - prev_model (optional): Pretrained model to continue training
        """
        if self.task == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=self.params['n_estimators'],
                max_depth=self.params['max_depth'],
                min_samples_split=self.params['min_samples_split'],
                min_samples_leaf=self.params['min_samples_leaf'],
                max_features=self.params['max_features'],
                random_state=self.params['random_state']
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=self.params['n_estimators'],
                max_depth=self.params['max_depth'],
                min_samples_split=self.params['min_samples_split'],
                min_samples_leaf=self.params['min_samples_leaf'],
                max_features=self.params['max_features'],
                random_state=self.params['random_state']
            )
        
        # If previous model exists, fit on top of it
        if prev_model is not None:
            self.model = prev_model
        
        self.model.fit(X, y)
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
        - X (np.array): Input features
        
        Returns:
        - Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def saveModel(self, path_to_save):
        """
        Save model to a pickle file
        
        Args:
        - path_to_save (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        with open(path_to_save, 'wb') as f:
            pickle.dump(self.model, f)

# Example usage
def main():
    # Create a Random Forest Classifier
    rf_clf = MLModel(task='classification')
    
    # Customize parameters
    rf_clf.set_params(
        n_estimators=200, 
        max_depth=7, 
        min_samples_split=5
    )
    
    # Create a Random Forest Regressor
    rf_reg = MLModel(task='regression')
    
    # Customize parameters
    rf_reg.set_params(
        max_features='sqrt', 
        min_samples_leaf=3
    )

if __name__ == "__main__":
    main()