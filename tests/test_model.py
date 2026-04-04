# test_model.py

import unittest
import mlflow
import os
import pandas as pd
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_tracking_uri("https://dagshub.com/saishkhandekar13/ML-MLOps-Project.mlflow")

        cls.client = mlflow.tracking.MlflowClient()
        cls.model_name = "my_model"

        versions = cls.client.get_latest_versions(cls.model_name)
        versions = [v for v in versions if v.current_stage == "Production"]

        if not versions:
            raise Exception("No model found in PRODUCTION stage")

        cls.model_version = versions[0].version
        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"

        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        print(f"Loaded model version: {cls.model_version}")

        with open("models/vectorizer.pkl", "rb") as f:
            cls.vectorizer = pickle.load(f)
        cls.test_data = pd.read_csv("data/processed/test_tfidf.csv")

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model)

    def test_feature_consistency(self):

        X_test = self.test_data.iloc[:, :-1]

        self.assertEqual(
            X_test.shape[1],
            self.model.metadata.get_input_schema().inputs[0].shape[-1]
            if hasattr(self.model, "metadata") else X_test.shape[1]
        )

    def test_model_signature(self):

        input_text = "this movie is amazing"

        features = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(features.toarray().astype("float64"))

        prediction = self.model.predict(input_df)

        self.assertEqual(len(prediction), 1)

    def test_model_performance(self):

        X_test = self.test_data.iloc[:, :-1]
        y_test = self.test_data.iloc[:, -1]

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\nAccuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        self.assertGreaterEqual(accuracy, 0.75)
        self.assertGreaterEqual(precision, 0.70)
        self.assertGreaterEqual(recall, 0.70)
        self.assertGreaterEqual(f1, 0.70)


if __name__ == "__main__":
    unittest.main()