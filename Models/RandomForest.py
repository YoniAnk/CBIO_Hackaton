import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

class MLModel:
    def __init__(self, model_type='rf', task='classification'):
        """
        Initialize the model with configurable parameters
        
        Args:
        - model_type (str): 'rf' for Random Forest or 'xgb' for XGBoost
        - task (str): 'classification' or 'regression'
        """
        # Global hyperparameters with default values
        self.params = {
            # Shared parameters
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
            
            # Random Forest specific
            'rf_min_samples_split': 2,
            'rf_min_samples_leaf': 1,
            
            # XGBoost specific
            'xgb_subsample': 0.8,
            'xgb_colsample_bytree': 0.8
        }
        
        self.model_type = model_type
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
        if self.model_type == 'rf':
            if self.task == 'classification':
                self.model = RandomForestClassifier(
                    n_estimators=self.params['n_estimators'],
                    max_depth=self.params['max_depth'],
                    min_samples_split=self.params['rf_min_samples_split'],
                    min_samples_leaf=self.params['rf_min_samples_leaf'],
                    random_state=self.params['random_state']
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=self.params['n_estimators'],
                    max_depth=self.params['max_depth'],
                    min_samples_split=self.params['rf_min_samples_split'],
                    min_samples_leaf=self.params['rf_min_samples_leaf'],
                    random_state=self.params['random_state']
                )
        
        elif self.model_type == 'xgb':
            if self.task == 'classification':
                self.model = XGBClassifier(
                    n_estimators=self.params['n_estimators'],
                    max_depth=self.params['max_depth'],
                    learning_rate=self.params['learning_rate'],
                    subsample=self.params['xgb_subsample'],
                    colsample_bytree=self.params['xgb_colsample_bytree'],
                    random_state=self.params['random_state']
                )
            else:
                self.model = XGBRegressor(
                    n_estimators=self.params['n_estimators'],
                    max_depth=self.params['max_depth'],
                    learning_rate=self.params['learning_rate'],
                    subsample=self.params['xgb_subsample'],
                    colsample_bytree=self.params['xgb_colsample_bytree'],
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
    rf_clf = MLModel(model_type='rf', task='classification')
    
    # Customize parameters
    rf_clf.set_params(
        n_estimators=200, 
        max_depth=7, 
        rf_min_samples_split=5
    )
    
    # Create a XGBoost Regressor
    xgb_reg = MLModel(model_type='xgb', task='regression')
    
    # Customize parameters
    xgb_reg.set_params(
        learning_rate=0.05, 
        xgb_subsample=0.9
    )

if __name__ == "__main__":
    main()