import numpy as np
from sklearn.model_selection import train_test_split

def load_and_split_tsv(file_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Load a TSV file, process RNA-seq data and labels, and split it into train, validation, and test sets
    while maintaining the original label ratios.

    Args:
        file_path (str): Path to the TSV file containing the dataset.
        train_ratio (float): Proportion of the data to use for the training set (default is 0.8).
        val_ratio (float): Proportion of the data to use for the validation set (default is 0.1).
        test_ratio (float): Proportion of the data to use for the test set (default is 0.1).

    Returns:
        tuple: A tuple containing three NumPy arrays:
            - train_data: Combined RNA-seq data and labels for the training set.
            - val_data: Combined RNA-seq data and labels for the validation set.
            - test_data: Combined RNA-seq data and labels for the test set.
    """
    # Load the full TSV file as strings, skipping the header
    full_data = np.genfromtxt(file_path, delimiter='\t', dtype=str, encoding='utf-8', skip_header=1)

    # Extract RNA-seq data (middle columns) and labels (last column)
    x_data = full_data[:, 1:-1].astype(float)  # Convert RNA-seq data to float
    y_data = full_data[:, -1].astype(int)      # Convert labels to int

    # First split: Train set (80%) and remaining set (20%), stratifying by labels
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_data,
        y_data,
        test_size=(1 - train_ratio),
        random_state=random_state,
        stratify=y_data  # Maintain label ratio
    )

    # Second split: Validation set (10%) and Test set (10%) from the remaining data
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=test_ratio / (test_ratio + val_ratio),  # Adjust split ratio for remaining data
        random_state=random_state,
        stratify=y_temp  # Maintain label ratio
    )

    # Combine RNA-seq data and labels for each set
    train_data = np.column_stack((x_train, y_train))
    val_data = np.column_stack((x_val, y_val))
    test_data = np.column_stack((x_test, y_test))

    return train_data, val_data, test_data


def split_xy(xy_data):
    """
    Split a combined dataset into RNA-seq data (X) and labels (Y).

    Args:
        xy_data (np.ndarray): Combined dataset with RNA-seq data and labels.

    Returns:
        tuple: A tuple containing:
            - x_data: RNA-seq data (all columns except the last).
            - y_data: Labels (last column).
    """
    x_data = xy_data[:, :-1]  # All columns except the last
    y_data = xy_data[:, -1]  # Only the last column
    return x_data, y_data