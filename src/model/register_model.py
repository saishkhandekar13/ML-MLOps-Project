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

mlflow.set_tracking_uri(
    "https://dagshub.com/saishkhandekar13/ML-MLOps-Project.mlflow"
)


# ================= Load =================
def load_model_info(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        model_info = json.load(file)
    logging.info('Model info loaded')
    return model_info


# ================= Main =================
def main():
    try:
        model_info = load_model_info('reports/experiment_info.json')
        run_id = model_info["run_id"]

        # 🔥 MUST MATCH model_evaluation logging name
        artifact_path = "logistic_regression_model"

        model_uri = f"runs:/{run_id}/{artifact_path}"
        model_name = "my_model"

        print(f"Registering model from: {model_uri}")

        # Register model
        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()

        # 🔥 Move to Production (CRITICAL FOR TEST)
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production"
        )

        print(f"✅ Model '{model_name}' version {model_version.version} is now in PRODUCTION")

    except Exception as e:
        logging.error('Failed to complete model registration: %s', e)
        print(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()