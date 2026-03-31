import numpy as np
import pandas as pd
import pickle
import os

from sklearn.linear_model import LogisticRegression
from src.logger import logging


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    logging.info('Data loaded from %s', file_path)
    return df


def train_model(X_train, y_train):

    model = LogisticRegression(
        C=0.1,
        solver='lbfgs',
        penalty='l2',
        class_weight='balanced',
        max_iter=3000
    )

    model.fit(X_train, y_train)
    logging.info("Model training completed")

    return model


def save_model(model, file_path: str):
    os.makedirs("models", exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info("Model saved to %s", file_path)


def main():
    train_data = load_data('./data/processed/train_tfidf.csv')

    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    model = train_model(X_train, y_train)

    save_model(model, 'models/model.pkl')


if __name__ == "__main__":
    main()
