"""Income prediction service.

Accepts a one-hot encoded census applicant, scores the saved XGBoost model,
stores the call, and forwards a copy to Evidently for drift metrics.

MODEL_SOURCE=local (the Docker default) loads ``models/clf.bin`` so the
service starts without a tracking server. MODEL_SOURCE=mlflow loads the
pyfunc artifact ``runs:/<MODEL_RUN_ID>/models_mlflow`` from
MLFLOW_TRACKING_URI.
"""

import logging
import os
import pickle
from datetime import datetime, timezone

import mlflow
import requests
import xgboost  # pylint: disable=unused-import
from flask import Flask, jsonify, request
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Feature names that DictVectorizer treated as numbers at training time.
# String values (for example "45") are coerced, because a string would be
# dropped and the model would score that field as zero.
NUMERIC_FEATURES = (
    "age",
    "financialWeight",
    "educationNum",
    "capitalGain",
    "capitalLoss",
    "hoursPerWeek",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("income_class")


def _load_artifacts():
    """Load the DictVectorizer and the classifier.

    Paths default to files next to this module so the process does not
    depend on the caller's working directory.
    """
    source = os.getenv("MODEL_SOURCE", "local").strip().lower()
    preprocessor_path = os.getenv(
        "PREPROCESSOR_PATH", os.path.join(BASE_DIR, "models", "preprocessor.b")
    )
    with open(preprocessor_path, "rb") as handle:
        vectorizer = pickle.load(handle)

    if source == "local":
        model_path = os.getenv(
            "LOCAL_MODEL_PATH", os.path.join(BASE_DIR, "models", "clf.bin")
        )
        with open(model_path, "rb") as handle:
            classifier = pickle.load(handle)
        version = os.getenv("MODEL_VERSION", "local-clf.bin")
        LOGGER.info("Loaded local model %s (%s)", model_path, version)
        return vectorizer, classifier, version

    if source == "mlflow":
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://10.138.0.5:5000")
        run_id = os.getenv("MODEL_RUN_ID")
        if not run_id:
            raise RuntimeError("MODEL_RUN_ID is required when MODEL_SOURCE=mlflow")
        mlflow.set_tracking_uri(tracking_uri)
        classifier = mlflow.pyfunc.load_model(f"runs:/{run_id}/models_mlflow")
        version = os.getenv("MODEL_VERSION", run_id)
        LOGGER.info("Loaded MLflow model run %s from %s", version, tracking_uri)
        return vectorizer, classifier, version

    raise RuntimeError(f"Unsupported MODEL_SOURCE '{source}'. Use 'local' or 'mlflow'.")


def _first_number(values):
    """Return the first prediction from an ndarray, list, or pandas object."""
    if hasattr(values, "iloc"):
        row = values.iloc[0]
        if hasattr(row, "iloc"):
            return float(row.iloc[0])
        return float(row)
    value = values[0]
    if isinstance(value, (list, tuple)):
        return float(value[0])
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _coerce_features(data):
    """Copy ``data`` and cast known numeric fields to float."""
    cleaned = dict(data)
    for name in NUMERIC_FEATURES:
        if name not in cleaned or cleaned[name] is None or cleaned[name] == "":
            continue
        try:
            cleaned[name] = float(cleaned[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{name}' must be a number") from exc
    return cleaned


def _jsonable_document(document):
    """Turn a Mongo document into plain JSON values."""
    payload = {}
    for key, value in document.items():
        if key == "_id":
            payload["id"] = str(value)
            continue
        if hasattr(value, "item"):
            value = value.item()
        payload[key] = value
    return payload


VECTORIZER, MODEL, MODEL_VERSION = _load_artifacts()

EVIDENTLY_SERVICE_ADDRESS = os.getenv("EVIDENTLY_SERVICE", "http://127.0.0.1:8085")
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27018")
EVIDENTLY_TIMEOUT_SECONDS = float(os.getenv("EVIDENTLY_TIMEOUT_SECONDS", "3"))

app = Flask("income_class")
mongo_client = MongoClient(MONGODB_ADDRESS, serverSelectionTimeoutMS=800)
db = mongo_client.get_database("prediction-service")
collection = db.get_collection("data")


def predict(data):
    """Score one applicant record. Returns the class label and P(income > $50k)."""
    features = VECTORIZER.transform(data)
    label = _first_number(MODEL.predict(features))
    probability = None
    if hasattr(MODEL, "predict_proba"):
        probabilities = MODEL.predict_proba(features)
        if hasattr(probabilities, "iloc"):
            probability = float(probabilities.iloc[0, 1])
        else:
            probability = float(probabilities[0][1])
    return label, probability


@app.route("/health", methods=["GET"])
def health():
    """Liveness probe used by Docker Compose and the web app."""
    return jsonify({"status": "ok", "model_version": MODEL_VERSION})


@app.route("/predict", methods=["POST"])
def predict_income():
    """Score a one-hot encoded applicant.

    The body is a JSON object whose keys match the training DictVectorizer
    vocabulary, for example ``{"age": 45, "education_ Bachelors": 1}``.
    The leading space in categorical keys is part of the census encoding.
    """
    data_point = request.get_json(silent=True)
    if not isinstance(data_point, dict):
        return (
            jsonify({"error": "Request body must be a JSON object of model features."}),
            400,
        )

    try:
        features = _coerce_features(data_point)
        label, probability = predict(features)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Prediction failed")
        return (
            jsonify({"error": "Could not score this applicant.", "detail": str(exc)}),
            400,
        )

    result = {
        "income_class": label,
        "model_version": MODEL_VERSION,
    }
    if label == 0:
        result["message"] = "Individual has annual income less than $50,000"
    else:
        result["message"] = "Individual has annual income greater than $50,000"
    if probability is not None:
        result["probability_over_50k"] = probability

    # Monitoring and history are best-effort. A down Mongo or Evidently
    # process should not hide a score the model already produced.
    save_to_db(features, label, probability)
    save_to_evidently_service(features, label)
    return jsonify(result)


@app.route("/predictions", methods=["GET"])
def recent_predictions():
    """Return the newest stored predictions, newest first."""
    try:
        limit = int(request.args.get("limit", 20))
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400
    limit = min(max(limit, 1), 100)

    try:
        cursor = collection.find().sort("_id", -1).limit(limit)
        return jsonify([_jsonable_document(document) for document in cursor])
    except PyMongoError as exc:
        LOGGER.warning("Could not read prediction history: %s", exc)
        return jsonify({"error": "prediction store unavailable"}), 503


def save_to_db(record, prediction, probability):
    """Persist the scored record. Failures are logged and swallowed."""
    stored = record.copy()
    stored["prediction"] = prediction
    stored["predicted_at"] = datetime.now(timezone.utc).isoformat()
    if probability is not None:
        stored["probability_over_50k"] = probability
    try:
        collection.insert_one(stored)
    except PyMongoError as exc:
        LOGGER.warning("Could not store prediction: %s", exc)


def save_to_evidently_service(record, prediction):
    """Forward the scored row to Evidently. Failures are logged and swallowed."""
    forwarded = record.copy()
    forwarded["prediction"] = prediction
    try:
        response = requests.post(
            f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/census",
            json=[forwarded],
            timeout=EVIDENTLY_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.warning("Evidently iterate failed: %s", exc)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
