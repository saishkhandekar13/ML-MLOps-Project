import json
import mlflow
from src.logger import logging
import os
import dagshub
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


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


# ================= Load =================
def load_model_info(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        model_info = json.load(file)
    logging.info('Model info loaded')
    return model_info


# ================= Register =================
def register_model(model_name: str, model_uri: str):
    try:
        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logging.info(f'{model_name} registered successfully')

    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise


# ================= Main =================
def main():
    try:
        model_info = load_model_info('reports/experiment_info.json')
        run_id = model_info["run_id"]

        # 🔥 Register BOTH models
        models = {
            "logistic_model": "logistic_regression_model",
            "svm_model": "svm_model"
        }

        for model_name, artifact_path in models.items():
            model_uri = f"runs:/{run_id}/{artifact_path}"
            register_model(model_name, model_uri)

    except Exception as e:
        logging.error('Failed to complete model registration: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()