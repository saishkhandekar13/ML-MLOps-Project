import os
import re
import string
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings("ignore")

# ========================== CONFIG ==========================
MLFLOW_TRACKING_URI = "https://dagshub.com/saishkhandekar13/ML-MLOps-Project.mlflow"

dagshub.init(
    repo_owner="saishkhandekar13",
    repo_name="ML-MLOps-Project",
    mlflow=True
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Logistic Regression Hyperparameter Tuning")


# ========================== PREPROCESSING ==========================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words).strip()


# ========================== LOAD DATA ==========================
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)

    df["review"] = df["review"].astype(str).apply(preprocess_text)

    df = df[df["sentiment"].isin(["positive", "negative"])]
    df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1})

    df = df.dropna().reset_index(drop=True)

    # TF-IDF Vectorization (Improved)
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X = vectorizer.fit_transform(df["review"])
    y = df["sentiment"]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    ), vectorizer, len(df)


# ========================== TRAINING ==========================
def train_and_log_model(X_train, X_test, y_train, y_test, vectorizer, dataset_size):

    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["liblinear", "lbfgs"]
    }

    with mlflow.start_run():

        # Log experiment info
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("max_features", 10000)
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_param("dataset_size", dataset_size)

        # Grid Search
        grid_search = GridSearchCV(
            LogisticRegression(max_iter=2000),
            param_grid,
            cv=3,
            scoring="f1",
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Predictions
        y_pred = best_model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        mlflow.log_metric("true_positive", cm[1][1])
        mlflow.log_metric("true_negative", cm[0][0])

        # Save vectorizer
        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
        mlflow.log_artifact("tfidf_vectorizer.pkl")

        # Log model
        mlflow.sklearn.log_model(
            best_model,
            "model",
            input_example=X_test[:5].toarray()
        )

        print("\n===== FINAL RESULTS =====")
        print(f"Best Params: {best_params}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")


# ========================== MAIN ==========================
if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), vectorizer, dataset_size = load_and_prepare_data("data/raw/IMDB.csv")
    train_and_log_model(X_train, X_test, y_train, y_test, vectorizer, dataset_size)