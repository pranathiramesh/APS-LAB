import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data():
    # load the dataset
    data = load_breast_cancer()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    y = pd.Series(
        (data.target == 0).astype(int),
        name="malignant"
    )

    print(y.value_counts())

    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print("Class names:", data.target_names)

    return X, y, data