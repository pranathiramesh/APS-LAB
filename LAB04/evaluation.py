import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def create_results(y_test, probabilities):
    # Store actual class and predicted probabilities
    results = pd.DataFrame({
        "Actual_class": y_test.values,
        "P_malignant": probabilities[:, 0],
        "P_benign": probabilities[:, 1]
    })

    print(results.head(10))

    return results


def add_actual_labels(results):
    # Convert actual class numbers into labels
    results["Actual_label"] = results["Actual_class"].map({
        0: "Malignant",
        1: "Benign"
    })

    print(
        results[
            ["Actual_label", "P_malignant", "P_benign"]
        ].head(10)
    )

    return results


def threshold_prediction(results, threshold=0.50):
    # Predict malignant if probability is greater than or equal
    # to the selected threshold
    results["Predicted_malignant"] = (
        results["P_malignant"] >= threshold
    ).astype(int)

    results["Predicted_label"] = results[
        "Predicted_malignant"
    ].map({
        1: "Malignant",
        0: "Benign"
    })

    print(
        results[
            ["Actual_class", "P_malignant", "Predicted_label"]
        ].head(10)
    )

    return results


def compare_thresholds(results):
    # Compare different thresholds
    for threshold in [0.30, 0.50, 0.70]:

        predictions = (
            results["P_malignant"] >= threshold
        ).astype(int)

        print(
            f"Threshold = {threshold}: "
            f"Predicted malignant cases = {predictions.sum()}"
        )


def confusion_matrices(y_test, probabilities):
    # Construct confusion matrices
    actual_malignant = (
        y_test.values == 0
    ).astype(int)

    for threshold in [0.30, 0.50, 0.70]:

        predicted_malignant = (
            probabilities[:, 0] >= threshold
        ).astype(int)

        cm = confusion_matrix(
            actual_malignant,
            predicted_malignant
        )

        print(f"\nThreshold = {threshold}")
        print(cm)


def compare_classification_metrics(y_test, probabilities):
    # Compare all metrics across different thresholds

    actual_malignant = (
        y_test.values == 0
    ).astype(int)

    metric_results = []

    for threshold in [0.10, 0.30, 0.50, 0.70, 0.90]:

        y_pred_threshold = (
            probabilities[:, 1] >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred_threshold,
            labels=[0, 1]
        ).ravel()

        accuracy = (
            (tp + tn) /
            (tp + tn + fp + fn)
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0
        )

        sensitivity_recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0
        )

        f1 = (
            2 * precision * sensitivity_recall /
            (precision + sensitivity_recall)
            if (precision + sensitivity_recall) > 0
            else 0
        )

        metric_results.append({
            "Threshold": threshold,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "Accuracy": accuracy,
            "Precision": precision,
            "Sensitivity_Recall": sensitivity_recall,
            "Specificity": specificity,
            "F1_Score": f1
        })

    threshold_metrics = pd.DataFrame(
        metric_results
    )

    print(threshold_metrics.round(3))

    return threshold_metrics