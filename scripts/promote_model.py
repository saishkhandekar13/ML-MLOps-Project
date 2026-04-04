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

        mlflow.set_tracking_uri(
            "https://dagshub.com/saishkhandekar13/ML-MLOps-Project.mlflow"
        )

        client = MlflowClient()

        model_name = "my_model"

        # ================= Get All Versions =================
        versions = client.get_latest_versions(model_name)

        if not versions:
            raise Exception("No model versions found")

        # 👉 Get latest version (highest version number)
        latest_version = max(versions, key=lambda v: int(v.version)).version

        print(f"Latest Model Version Found: {latest_version}")

        # ================= Archive Existing Production =================
        prod_versions = [v for v in versions if v.current_stage == "Production"]

        for version in prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )
            print(f"Archived old Production model: {version.version}")

        # ================= Promote Latest to Production =================
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Production"
        )

        print(f"✅ Model version {latest_version} promoted to PRODUCTION")

    except Exception as e:
        print(f"❌ Error in model promotion: {e}")
        raise


if __name__ == "__main__":
    promote_model()