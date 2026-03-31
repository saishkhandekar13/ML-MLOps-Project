from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub
import numpy as np

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------------------
# TEXT PREPROCESSING (MATCHED WITH PIPELINE)
# ------------------------------------------------------------------------------------------

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

def remove_stop_words(text):
    return " ".join([word for word in text.split() if word not in stop_words])

def lemmatization(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def normalize_text(text):
    text = text.lower()
    text = removing_urls(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = remove_stop_words(text)
    text = lemmatization(text)
    return text

# ------------------------------------------------------------------------------------------
# DAGSHUB + MLFLOW SETUP (PRODUCTION)
# ------------------------------------------------------------------------------------------

dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "saishkhandekar13"
repo_name = "ML-MLOps-Project"   # ✅ FIXED

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# ------------------------------------------------------------------------------------------
# FLASK APP INITIALIZATION
# ------------------------------------------------------------------------------------------

app = Flask(__name__)

# ------------------------------------------------------------------------------------------
# PROMETHEUS METRICS
# ------------------------------------------------------------------------------------------

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests", ["method", "endpoint"], registry=registry
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Request latency", ["endpoint"], registry=registry
)

PREDICTION_COUNT = Counter(
    "model_prediction_count", "Prediction count", ["prediction"], registry=registry
)

# ------------------------------------------------------------------------------------------
# MODEL + VECTORIZER LOADING
# ------------------------------------------------------------------------------------------

model_name = "my_model"
model_uri = f"models:/{model_name}/latest"

print(f"Loading model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)

vectorizer_path = os.path.join("models", "vectorizer.pkl")
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# ------------------------------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------------------------------

@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()

    response = render_template("index.html", result=None)

    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response


@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    text = request.form["text"]

    # ✅ Preprocess
    text = normalize_text(text)

    # ✅ Vectorize
    features = vectorizer.transform([text])
    features_df = pd.DataFrame(features.toarray())

    # ✅ Predict
    prediction = model.predict(features_df)[0]

    # Metrics
    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template("index.html", result=prediction)


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


# ------------------------------------------------------------------------------------------
# RUN APP
# ------------------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)