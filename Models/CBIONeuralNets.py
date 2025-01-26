################################# Imports #####################################
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import tqdm as tqdm
import matplotlib.pyplot as plt
################################# Constants ###################################
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"
################################# Classes #####################################

class CBIONN(nn.Module):
    def __init__(self, input_size, num_classes, dropout=False):
        super(CBIONN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.fc1 = nn.Linear(self.input_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, self.num_classes)
        self.relu = nn.ReLU()
        self.dropout02 = nn.Dropout(0.2)
        self.dropout03 = nn.Dropout(0.3)
        self.dropout05 = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)

        if self.dropout:
            x = self.dropout02(x)

        x = self.fc2(x)
        x = self.relu(x)

        if self.dropout:
            x = self.dropout03(x)

        x = self.fc3(x)
        x = self.relu(x)

        if self.dropout:
            x = self.dropout05(x)

        x = self.fc4(x)
        x = self.relu(x)

        x = self.fc5(x)
        x = self.relu(x)

        x = self.fc6(x)

        return x

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        x = torch.tensor(X, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            outputs = self.forward(x)
            outputs = self.softmax(outputs)
            _, y_pred = torch.max(outputs, 1)

        return y_pred.cpu().numpy()

    def saveModel(self, path_to_save: str) -> None:
        """
        Saves the trained model as a pickle file.

        Parameters:
        - path_to_save: File path to save the model.

        Returns:
        - None
        """
        with open(path_to_save, 'wb') as f:
            pickle.dump(self, f)

    def train_model(self, X: np.array, y: np.array, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LEARNING_RATE) -> None:
        """
        Trains the neural network.

        Parameters:
        - X: Training features as a numpy array.
        - y: Training labels as a numpy array.

        Returns:
        - None
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y, dtype=torch.long).squeeze().to(DEVICE)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train()

        losses = []

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")
            losses.append(running_loss / len(train_loader))

        # plot_losses(losses)

class CBIOCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        """
        Initializes the CNN for 1D input.

        Parameters:
        - input_dim: Number of input features (genes).
        - num_classes: Number of output classes (levels of cancer severity).
        """
        super(CBIOCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear((input_dim // 8) * 64, 128)  # Adjust dimensions based on pooling
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass of the CNN.

        Parameters:
        - x: Input tensor (batch_size, input_dim).

        Returns:
        - Output tensor.
        """
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, input_dim)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions on new data.

        Parameters:
        - X: Input features as a numpy array.

        Returns:
        - Predicted class labels as a numpy array.
        """
        self.eval()
        x = torch.tensor(X, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            outputs = self.forward(x)
            outputs = self.softmax(outputs)
            _, y_pred = torch.max(outputs, 1)

        return y_pred.cpu().numpy()

    def saveModel(self, path_to_save: str) -> None:
        """
        Saves the trained model as a pickle file.

        Parameters:
        - path_to_save: File path to save the model.

        Returns:
        - None
        """
        with open(path_to_save, 'wb') as f:
            pickle.dump(self, f)

    def train_model(self, X: np.array, y: np.array, batch_size=64, epochs=10, lr=0.001) -> None:
        """
        Trains the CNN.

        Parameters:
        - X: Training features as a numpy array.
        - y: Training labels as a numpy array.
        - batch_size: Batch size for training.
        - epochs: Number of training epochs.
        - lr: Learning rate for the optimizer.
        - optimizer_type: Type of optimizer to use ("adam" or "sgd").

        Returns:
        - None
        """
        criterion = nn.CrossEntropyLoss()


        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y, dtype=torch.long).squeeze().to(DEVICE)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

def plot_losses(losses, save=False, show=True, save_path='loss_plot.png'):
    """
    Plots the training loss as a function of epochs.

    Parameters:
    - losses: List of loss values for each epoch.
    - save: Boolean flag to save the plot to a file.
    - show: Boolean flag to display the plot.
    - save_path: File path to save the plot if save is True.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss as a Function of Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)

    if save:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()