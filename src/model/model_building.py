import numpy as np
import pandas as pd
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from src.logger import logging


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    logging.info('Data loaded from %s', file_path)
    return df


def train_model(X_train, y_train):

    models = {
        "logistic_regression": LogisticRegression(
            C=2,
            solver='saga',
            n_jobs=-1,
            max_iter=1500
        ),
        "svm": LinearSVC(C=1.5)
    }

    trained_models = {}

    for name, m in models.items():
        m.fit(X_train, y_train)
        print(f"{name} trained")
        trained_models[name] = m

    logging.info("Model training completed")

    return trained_models


def save_model(model, file_path: str):
    os.makedirs("models", exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info("Model saved to %s", file_path)


def main():
    train_data = load_data('./data/processed/train_tfidf.csv')

    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    models = train_model(X_train, y_train)

    # Save both models separately
    save_model(models["logistic_regression"], 'models/logistic_model.pkl')
    save_model(models["svm"], 'models/svm_model.pkl')


if __name__ == "__main__":
    main()