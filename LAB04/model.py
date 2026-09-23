from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def train_model(X, y):
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training size:", len(y_train))
    print("Testing size:", len(y_test))

    print("\nTraining proportions:")
    print(y_train.value_counts(normalize=True).sort_index())

    print("\nTesting proportions:")
    print(y_test.value_counts(normalize=True).sort_index())

    # create the Logistic Regression model
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    # train the model
    model.fit(X_train, y_train)

    # get predicted probabilities
    probabilities = model.predict_proba(X_test)

    print("Model classes:", model.named_steps["logisticregression"].classes_)

    print("\nFirst 5 predicted probabilities:")
    print(probabilities[:5])

    return (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        probabilities
    )