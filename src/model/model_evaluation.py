import numpy as np
import pandas as pd
import pickle
import json
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import mlflow
import mlflow.sklearn
import dagshub

from src.logger import logging


# ================= MLflow Setup =================
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "saishkhandekar13"
repo_name = "ML-MLOps-Project"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


# ================= Load Functions =================
def load_model(file_path: str):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    logging.info(f'Model loaded from {file_path}')
    return model


def load_data(file_path: str):
    df = pd.read_csv(file_path)
    logging.info(f'Data loaded from {file_path}')
    return df


# ================= Evaluation =================
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = roc_auc_score(y_test, y_pred)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": auc
    }

    logging.info("Evaluation completed")
    return metrics


# ================= Save =================
def save_metrics(metrics, path):
    os.makedirs("reports", exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def save_model_info(run_id, path):
    info = {
        "run_id": run_id,
        "model_path": "model"
    }
    with open(path, "w") as f:
        json.dump(info, f, indent=4)


# ================= Main =================
def main():

    mlflow.set_experiment("my-dvc-pipeline")

    with mlflow.start_run() as run:

        # Load correct model + TFIDF data
        model = load_model("models/model.pkl")
        test_data = load_data("data/processed/test_tfidf.csv")

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        # 🚨 IMPORTANT CHECK (prevents your error forever)
        if X_test.shape[1] != model.n_features_in_:
            raise ValueError(
                f"Feature mismatch: Data has {X_test.shape[1]} features, "
                f"model expects {model.n_features_in_}"
            )

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Save metrics
        save_metrics(metrics, "reports/metrics.json")

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log params
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        # ✅ CRITICAL FIX: log model with input example
        mlflow.sklearn.log_model(
            model,
            "model",
            input_example=X_test[:5]
        )

        # Save model info
        save_model_info(run.info.run_id, "reports/experiment_info.json")

        mlflow.log_artifact("reports/metrics.json")

        logging.info("Model evaluation + logging completed successfully")


if __name__ == "__main__":
    main()