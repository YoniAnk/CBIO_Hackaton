import numpy as np
from networkx.classes import non_neighbors
# import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import time
from Data.load_and_split_data import load_and_split_tsv, split_xy, balance_dataset
from Models.CBIONeuralNets import CBIONN, CBIOCNN

# Import our custom models
from Models.KNNClassifier import KNNClassifier
# from Models.CBIONeuralNets import CBIONN, CBIOCNN
from Models.LogisticRegression import LogisticRegression
from Utils.ModelEvaluation import ModelEvaluator


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
    # Set random seeds for reproducibility
    np.random.seed()
    # torch.manual_seed(42)

    for file_path in ["./Data/rna_seq_with_012_labels.tsv",
                      "./Data/rna_seq_only_sick_10_genes.tsv",
                      "./Data/rna_seq_only_sick_50_genes.tsv",
                      "./Data/rna_seq_only_sick_100_genes.tsv"
    ]:


        train_data, val_data, test_data = load_and_split_tsv(
            file_path, random_state=np.random.seed())

        for sampling_method in ["over", "under", "standard"]:

            if sampling_method == "standard":
                train_X, train_Y = split_xy(train_data)
            else:
                train_X, train_Y = split_xy(balance_dataset(train_data, sampling_method))

            val_X, val_Y = split_xy(val_data)
            test_X, test_Y = split_xy(test_data)

            n_features = train_X.shape[1]
            n_classes = len(np.unique(train_Y))

            # Initialize models
            n_neighbors = 5

            learning_rate = 0.001
            n_iteration = 1000
            batch_size = 16
            lambda_reg = 0.005

            models = {
                'KNN': KNNClassifier(n_neighbors=n_neighbors)
                ,
                'LogisticRegression': LogisticRegression(
                    learning_rate=learning_rate,
                    n_iterations=n_iteration,
                    batch_size=batch_size,
                    lambda_reg=lambda_reg
                )
            }

            # Train and evaluate each model
            evaluation_results = {}

            for model_name, model in models.items():
                print(f"\nTraining {model_name}...")
                start_time = time.time()

                model.train_model(train_X, train_Y)

                training_time = time.time() - start_time
                print(f"{model_name} training completed in {training_time:.2f} seconds")

                # Evaluate the model
                parameters = n_neighbors if model_name =='KNN' else (learning_rate, n_iteration, batch_size,lambda_reg)
                title = f"\n{model_name} parameters: {parameters}, \ndata sampling: {sampling_method} using {file_path}"
                evaluator = evaluate_model(model, title, test_X, test_Y)
                evaluation_results[model_name] = evaluator

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

                print(f"\n{model_name} , data sampling: {sampling_method}, parameters: {parameters}")
                print(f"Average F1 Score: {avg_f1:.3f}")
                print(f"Average Accuracy: {avg_accuracy:.3f}")


if __name__ == "__main__":
    main()