import os
import re
import string
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings("ignore")

# ========================== CONFIG ==========================
CONFIG = {
    "data_path": "data/raw/IMDB.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/saishkhandekar13/ML-MLOps-Project.mlflow",
    "repo_owner": "saishkhandekar13",
    "repo_name": "ML-MLOps-Project",
    "experiment_name": "Linear Models Comparison"
}

# ========================== SETUP ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(
    repo_owner=CONFIG["repo_owner"],
    repo_name=CONFIG["repo_name"],
    mlflow=True
)
mlflow.set_experiment(CONFIG["experiment_name"])

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

    return " ".join(words)

# ========================== LOAD DATA ==========================
def load_data(path):
    df = pd.read_csv(path)

    df["review"] = df["review"].astype(str).apply(preprocess_text)
    df = df[df["sentiment"].isin(["positive", "negative"])]
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

    df = df.dropna().reset_index(drop=True)

    return df

# ========================== VECTORIZATION ==========================
def vectorize(df):
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X = vectorizer.fit_transform(df["review"])
    y = df["sentiment"]

    return X, y, vectorizer

# ========================== MODELS ==========================
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "LinearSVC": LinearSVC(),
    "SGDClassifier": SGDClassifier(loss="log_loss")
}

# ========================== TRAIN ==========================
def train_models(X, y, vectorizer):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["test_size"],
        random_state=42,
        stratify=y
    )

    with mlflow.start_run(run_name="Linear Models"):

        for name, model in MODELS.items():

            with mlflow.start_run(run_name=name, nested=True):

                # Log params
                mlflow.log_param("model", name)
                mlflow.log_param("vectorizer", "TF-IDF")
                mlflow.log_param("max_features", 10000)
                mlflow.log_param("ngram_range", "(1,2)")
                mlflow.log_param("dataset_size", len(y))

                # Train
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)

                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

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

                # Save model
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    input_example=X_test[:5].toarray()
                )

                print(f"\n{name}")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"F1 Score: {f1:.4f}")

        # Save vectorizer once
        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
        mlflow.log_artifact("tfidf_vectorizer.pkl")

# ========================== MAIN ==========================
if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    X, y, vectorizer = vectorize(df)
    train_models(X, y, vectorizer)