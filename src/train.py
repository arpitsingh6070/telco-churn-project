import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from data_preprocessing import load_data, get_preprocessor
from sklearn.pipeline import Pipeline
import mlflow
import dagshub
import os
import joblib

def main():
    # Initialize DagsHub MLflow if running with credentials
    print("Setting up MLflow tracking...")
    # NOTE: To use DagsHub properly, you must replace 'USERNAME' and 'REPO_NAME'
    # with your actual DagsHub username and repository name.
    # We use a try-except block so it falls back to local tracking if not configured.
    try:
        # dagshub.init(repo_owner='YOUR_DAGSHUB_USERNAME', repo_name='YOUR_DAGSHUB_REPO_NAME', mlflow=True)
        pass # Commented out by default until you setup DagsHub repo
    except Exception as e:
        pass
    
    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    mlflow.set_experiment("Telco_Customer_Churn_Experiment")
    
    print("Loading data...")
    df = load_data('../WA_Fn-UseC_-Telco-Customer-Churn.csv')
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Defining pipeline...")
    preprocessor = get_preprocessor()
    model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    with mlflow.start_run():
        print("Training model...")
        clf.fit(X_train, y_train)
        
        print("Evaluating model...")
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("learning_rate", 0.1)
        
        # Log model
        mlflow.sklearn.log_model(clf, "model")
        
        # Save locally for API to use
        os.makedirs('../models', exist_ok=True)
        joblib.dump(clf, '../models/churn_model.pkl')
        print("Model saved to ../models/churn_model.pkl")

if __name__ == "__main__":
    main()
