# promote_model.py

import os
import mlflow
from mlflow.tracking import MlflowClient


def promote_model():
    try:
        # ================= MLflow Setup =================
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "saishkhandekar13"
        repo_name = "ML-MLOps-Project"   # ✅ UPDATED

        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        client = MlflowClient()

        model_name = "my_model"

        # ================= Get Staging Model =================
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])

        if not staging_versions:
            raise Exception("No model found in STAGING stage")

        latest_staging_version = staging_versions[0].version

        print(f"Found Staging Model Version: {latest_staging_version}")

        # ================= Archive Existing Production =================
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])

        if prod_versions:
            for version in prod_versions:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )
                print(f"Archived Production Model Version: {version.version}")
        else:
            print("No existing Production model found")

        # ================= Promote New Model =================
        client.transition_model_version_stage(
            name=model_name,
            version=latest_staging_version,
            stage="Production"
        )

        print(f"✅ Model version {latest_staging_version} promoted to PRODUCTION")

    except Exception as e:
        print(f"❌ Error in model promotion: {e}")
        raise


if __name__ == "__main__":
    promote_model()