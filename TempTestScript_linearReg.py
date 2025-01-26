import numpy as np
# import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import time

# Import our custom models
from Models.KNNClassifier import KNNClassifier
# from Models.CBIONeuralNets import CBIONN, CBIOCNN
from Models.LogisticRegression import LogisticRegression
from Utils.ModelEvaluation import ModelEvaluator
from Data.load_and_split_data import load_and_split_tsv
from Models.LinearRegressionModel import LinearRegressionModel

def generate_synthetic_data(n_samples=1000, n_classes=3, n_features=10):
    """
    Generate synthetic data for multi-class classification with clear class separation.
    The number of features is set to 10 to provide enough information for our neural models.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features - 2,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=2,
        class_sep=2.0,  # Increased class separation for clearer evaluation
        random_state=42
    )

    return X, y


def evaluate_model(model, model_name: str, X_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluate a single model using our enhanced ModelEvaluator.
    """
    print(f"\n{'=' * 20} Evaluating {model_name} {'=' * 20}")
    evaluator = ModelEvaluator(model, X_test, y_test)

    # Generate comprehensive evaluation report
    print(f"\nPerformance Metrics for {model_name}:")
    evaluator.generate_evaluation_report(model_name=model_name)

    return evaluator


def main():

    # # Set random seeds for reproducibility
    # np.random.seed(42)
    # # torch.manual_seed(42)

    # # Generate synthetic multi-class data
    # print("Generating synthetic data...")
    # n_classes = 3
    # n_features = 10
    # X, y = generate_synthetic_data(
    #     n_samples=1000,
    #     n_classes=n_classes,
    #     n_features=n_features
    # )
    train_data, val_data, test_data = load_and_split_tsv("rna_seq_with_012_labels.tsv")

    # Split the data
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_val, y_val = val_data[:, :-1], val_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    model_name = "LinearRegression"
    models =  LinearRegressionModel()
        # 'CBIONN': CBIONN(
        #     input_size=n_features,
        #     num_classes=n_classes,
        #     dropout=True
        # ),
        # 'CBIOCNN': CBIOCNN(
        #     input_dim=n_features,
        #     num_classes=n_classes
        # )
    # }
    # Train and evaluate each model
    evaluation_results = {}

    start_time = time.time()

    # Train the model
    models.train_model(X_train_scaled, y_train)

    training_time = time.time() - start_time
    print(f"{model_name} training completed in {training_time:.2f} seconds")

    # Evaluate the model
    evaluator = evaluate_model(models, model_name, X_test_scaled, y_test)
    # evaluator = models.predict(X_val)
    evaluation_results[model_name] = evaluator

    # Save the trained model
    model_path = f"trained_{model_name.lower()}_model.pkl"
    print(f"Saving {model_name} to {model_path}")
    # model.saveModel(model_path)

    # Compare model performances
    print("\n" + "=" * 50)
    print("Model Performance Comparison")
    print("=" * 50)

    for model_name, evaluator in evaluation_results.items():
        metrics = evaluator.calculate_class_metrics()
        avg_f1 = np.mean([class_metrics['F1 Score']
                            for class_metrics in metrics.values()])
        avg_accuracy = np.mean([class_metrics['Accuracy']
                                for class_metrics in metrics.values()])

        print(f"\n{model_name}:")
        print(f"Average F1 Score: {avg_f1:.3f}")
        print(f"Average Accuracy: {avg_accuracy:.3f}")


if __name__ == "__main__":
    main()