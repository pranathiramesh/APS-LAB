from data_loader import load_data
from model import train_model
from evaluation import (
    create_results,
    add_actual_labels,
    threshold_prediction,
    compare_thresholds,
    confusion_matrices,
    compare_classification_metrics
)


def main():

    # Load the Breast Cancer dataset
    X, y, data = load_data()

    # Train the Logistic Regression model
    (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        probabilities
    ) = train_model(X, y)

    # Create results table
    results = create_results(
        y_test,
        probabilities
    )

    # Add actual class labels
    results = add_actual_labels(results)

    # Predict using threshold 0.50
    results = threshold_prediction(
        results,
        threshold=0.50
    )

    # Compare different thresholds
    compare_thresholds(results)

    # Display confusion matrices
    confusion_matrices(
        y_test,
        probabilities
    )

    # Compare classification metrics
    threshold_metrics = compare_classification_metrics(
        y_test,
        probabilities
    )

    print("\nFinal threshold metrics:")
    print(threshold_metrics)


if __name__ == "__main__":
    main()