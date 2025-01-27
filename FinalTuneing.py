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
    np.random.seed(42)

    for file_path in ["./Data/rna_seq_with_012_labels.tsv",

                      ]:

        train_data, val_data, test_data = load_and_split_tsv(file_path)

        train_X, train_Y = split_xy(balance_dataset(train_data, "under"))

        val_X, val_Y = split_xy(val_data)
        test_X, test_Y = split_xy(test_data)

        n_features = train_X.shape[1]
        n_classes = len(np.unique(train_Y))

        learning_rates = [0.001, 0.005 ,0.01, 0.05 ,0.1]
        n_iterations = [500, 1000, 2000, 5000]
        batch_sizes = [8 ,16 ,32, 64]
        lambda_regs = [0.001, 0.005, 0.01, 0.05, 0.1]

        # List comprehension for all combinations
        param_grid = [(lr, n_iter, bs, lam)
                          for lr in learning_rates
                          for n_iter in n_iterations
                          for bs in batch_sizes
                          for lam in lambda_regs]

        for (lr, n_iter, bs, lam) in param_grid:

            models = {
                'LogisticRegression': LogisticRegression(
                    learning_rate=lr,
                    n_iterations=n_iter,
                    batch_size=bs,
                    lambda_reg=lam
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
                parameters = (lr, n_iter, bs, lam)
                title = f"\n{model_name} parameters: {parameters}, \ndata sampling: under using {file_path}"
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

                print(f"\n{model_name} , data sampling: under, parameters: {parameters}")
                print(f"Average F1 Score: {avg_f1:.3f}")
                print(f"Average Accuracy: {avg_accuracy:.3f}")


if __name__ == "__main__":
    main()