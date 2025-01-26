import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class LinearRegressionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, X, y, prev_model=None):
        """
        Train the linear regression model
        
        Parameters:
        X (np.array): Input features
        y (np.array): Target values
        prev_model (optional): Previous model to continue training
        """
        # Standardize the features
        X_scaled = self.scaler.fit_transform(X)
        
        # If prev_model is provided, use it; otherwise create a new model
        if prev_model is not None:
            self.model = prev_model
        else:
            self.model = LinearRegression()
        
        # Fit the model
        self.model.fit(X_scaled, y)
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Parameters:
        X (np.array): Input features for prediction
        
        Returns:
        np.array: Predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Standardize the input features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        return self.model.predict(X_scaled)
    
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
    


# Create model
model = LinearRegressionModel()

# Train the model
model.train(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Save the model
model.saveModel('my_linear_regression_model.pkl')

# Load the model later
loaded_model = LinearRegressionModel.loadModel('my_linear_regression_model.pkl')