import numpy as np
import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logging


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.fillna('', inplace=True)
    logging.info('Data loaded from %s', file_path)
    return df


def apply_tfidf(train_data, test_data):
    logging.info("Applying TF-IDF...")

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words='english'
    )

    X_train = train_data['review'].values
    y_train = train_data['sentiment'].values
    X_test = test_data['review'].values
    y_test = test_data['sentiment'].values

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    train_df = pd.DataFrame(X_train_vec.toarray())
    train_df['label'] = y_train

    test_df = pd.DataFrame(X_test_vec.toarray())
    test_df['label'] = y_test

    os.makedirs("models", exist_ok=True)
    pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))

    logging.info("TF-IDF transformation completed")

    return train_df, test_df


def main():
    train_data = load_data('./data/interim/train_processed.csv')
    test_data = load_data('./data/interim/test_processed.csv')

    train_df, test_df = apply_tfidf(train_data, test_data)

    os.makedirs("./data/processed", exist_ok=True)
    train_df.to_csv('./data/processed/train_tfidf.csv', index=False)
    test_df.to_csv('./data/processed/test_tfidf.csv', index=False)


if __name__ == "__main__":
    main()