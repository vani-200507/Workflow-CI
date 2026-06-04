import pandas as pd
import numpy as np
import os
import sys
import io

# Force UTF-8 encoding for stdout and stderr to prevent encoding errors with emojis on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(n_estimators, max_depth):
    # Set tracking URI to local tracking (will be overriden by MLflow if tracking URI is set via environment variable)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Loan_Approval_CI_Retraining")
    
    # Load preprocessing data
    data_path = os.path.join("namadataset_preprocessing", "loan_preprocessing.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "namadataset_preprocessing", "loan_preprocessing.csv")
        
    print(f"Loading preprocessed data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Split features and target
    X = df.drop(columns=['Loan_Status'])
    y = df['Loan_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Enable Autologging for Scikit-Learn
    mlflow.sklearn.autolog()
    
    # Start MLflow run
    with mlflow.start_run(run_name="CI_Retraining_Run"):
        print(f"Training Random Forest Classifier with n_estimators={n_estimators}, max_depth={max_depth}...")
        
        # Instantiate and train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained successfully! Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain loan approval model via MLflow Project.")
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the forest')
    parser.add_argument('--max_depth', type=int, default=5, help='Maximum depth of the tree')
    
    args = parser.parse_args()
    train_model(args.n_estimators, args.max_depth)
