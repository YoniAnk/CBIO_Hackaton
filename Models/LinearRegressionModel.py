import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from Models.AbstractModel import Model

class LinearRegressionModel(Model):
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def train_model(self, X, y, prev_model=None):
        """
        Train the linear regression model
        """
        X_scaled = self.scaler.fit_transform(X)

        if prev_model is not None:
            self.model = prev_model
        else:
            self.model = LinearRegression()

        self.model.fit(X_scaled, y)
       
    def predict(self, X):
        """
        Make predictions using the trained model, ensuring output is rounded to {0, 1, 2}.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        # Ensure predictions are in range {0, 1}
        predictions = np.clip(predictions, 0, 1)

        # Round predictions to nearest valid label
        possible_labels = np.array([0, 1])
        predictions = possible_labels[
            np.abs(predictions[:, None] - possible_labels).argmin(axis=1)]

        return predictions
    
    def saveModel(self, path_to_save):
        """
        Save the model to a pickle file
        
        Parameters:
        path_to_save (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Create a dictionary to save both model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        
        # Save the model and scaler
        with open(path_to_save, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def loadModel(cls, path_to_load):
        """
        Load a previously saved model
        
        Parameters:
        path_to_load (str): Path to load the model from
        
        Returns:
        LinearRegressionModel: Loaded model instance
        """
        with open(path_to_load, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance and set the loaded model and scaler
        instance = cls()
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        
        return instance
    
    def get_weights(self):
        return self.model.coef_, self.model.intercept_