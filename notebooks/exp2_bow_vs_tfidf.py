import os
import re
import string
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import joblib

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse

import warnings
warnings.filterwarnings("ignore")

# ========================== CONFIG ==========================
CONFIG = {
    "data_path": "data/raw/IMDB.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/saishkhandekar13/ML-MLOps-Project.mlflow",
    "dagshub_repo_owner": "saishkhandekar13",
    "dagshub_repo_name": "ML-MLOps-Project",
    "experiment_name": "BoW vs TF-IDF Comparison"
}

# ========================== SETUP ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(
    repo_owner=CONFIG["dagshub_repo_owner"],
    repo_name=CONFIG["dagshub_repo_name"],
    mlflow=True
)
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== PREPROCESSING ==========================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)
    text = re.sub(r'\d+', '', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['review'] = df['review'].apply(preprocess_text)

    df['sentiment'] = df['sentiment'].str.lower()
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    df = df.dropna()
    df = df.reset_index(drop=True)

    return df

# ========================== FEATURE ENGINEERING ==========================
VECTORIZERS = {
    'BoW': CountVectorizer(
        max_features=10000,
        ngram_range=(1,2),
        stop_words="english"
    ),
    'TF-IDF': TfidfVectorizer(
        max_features=10000,
        ngram_range=(1,2),
        stop_words="english"
    )
}

# ========================== MODELS ==========================
ALGORITHMS = {
    'LogisticRegression': LogisticRegression(max_iter=2000, solver="liblinear"),
    'MultinomialNB': MultinomialNB(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100)
}

# ========================== TRAINING ==========================
def train_and_evaluate(df):

    with mlflow.start_run(run_name="All Experiments"):

        for algo_name, algorithm in ALGORITHMS.items():
            for vec_name, vectorizer in VECTORIZERS.items():

                with mlflow.start_run(run_name=f"{algo_name} + {vec_name}", nested=True):

                    try:
                        # Feature extraction
                        X = vectorizer.fit_transform(df['review'])
                        y = df['sentiment']

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y,
                            test_size=CONFIG["test_size"],
                            random_state=42,
                            stratify=y
                        )

                        # Logging parameters
                        mlflow.log_params({
                            "vectorizer": vec_name,
                            "algorithm": algo_name,
                            "max_features": 10000,
                            "ngram_range": "(1,2)",
                            "test_size": CONFIG["test_size"],
                            "dataset_size": len(df)
                        })

                        # Train
                        model = algorithm
                        model.fit(X_train, y_train)

                        # Predictions
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

                        # Confusion matrix
                        cm = confusion_matrix(y_test, y_pred)
                        mlflow.log_metric("true_positive", cm[1][1])
                        mlflow.log_metric("true_negative", cm[0][0])

                        # Save vectorizer
                        vec_path = f"{vec_name}_vectorizer.pkl"
                        joblib.dump(vectorizer, vec_path)
                        mlflow.log_artifact(vec_path)

                        # Log model
                        input_example = X_test[:5] if not scipy.sparse.issparse(X_test) else X_test[:5].toarray()
                        mlflow.sklearn.log_model(model, "model", input_example=input_example)

                        print(f"\n{algo_name} + {vec_name}")
                        print(f"Accuracy: {accuracy:.4f}")

                    except Exception as e:
                        print(f"Error: {algo_name} + {vec_name} → {e}")
                        mlflow.log_param("error", str(e))


# ========================== RUN ==========================
if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    train_and_evaluate(df)