
import pandas as pd
import os
import argparse
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_model(n_estimators, max_depth):

    # Set experiment
    mlflow.set_experiment("Loan_Approval_CI_Retraining")

    # Load dataset
    data_path = os.path.join(
        "namadataset_preprocessing",
        "loan_preprocessing.csv"
    )

    if not os.path.exists(data_path):
        data_path = os.path.join(
            os.path.dirname(__file__),
            "namadataset_preprocessing",
            "loan_preprocessing.csv"
        )

    print(f"Loading preprocessed data from {data_path}...")

    df = pd.read_csv(data_path)

    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Enable MLflow autologging
    mlflow.sklearn.autolog()

    print(
        f"Training Random Forest Classifier "
        f"with n_estimators={n_estimators}, "
        f"max_depth={max_depth}"
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, "model")

    print(f"Model trained successfully! Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=5
    )

    args = parser.parse_args()

    train_model(
        args.n_estimators,
        args.max_depth
    )
