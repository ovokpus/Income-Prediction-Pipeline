"""API tests against the Flask app and the saved local model.

Mongo and Evidently are pointed at closed ports. Scoring must still succeed,
and history must report that the store is down instead of hanging.
"""

# isort: skip_file

import os
import sys

os.environ["MODEL_SOURCE"] = "local"
os.environ["MODEL_VERSION"] = "test-local"
os.environ["MONGODB_ADDRESS"] = "mongodb://127.0.0.1:1"
os.environ["EVIDENTLY_SERVICE"] = "http://127.0.0.1:9"
os.environ["EVIDENTLY_TIMEOUT_SECONDS"] = "1"

SERVICE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "prediction_service")
)
FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend")
)
sys.path.insert(0, SERVICE_DIR)
sys.path.insert(0, FRONTEND_DIR)

from features import PRESETS, encode_applicant  # pylint: disable=wrong-import-position
import predict  # pylint: disable=wrong-import-position


def test_health_reports_local_model():
    response = predict.app.test_client().get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["model_version"] == "test-local"


def test_predict_part_time_example_is_at_or_under_50k():
    payload = encode_applicant(PRESETS["part_time"])
    response = predict.app.test_client().post("/predict", json=payload)
    assert response.status_code == 200
    body = response.get_json()
    assert body["income_class"] == 0.0
    assert "less than $50,000" in body["message"]
    assert body["probability_over_50k"] < 0.5


def test_predict_manager_example_is_over_50k():
    payload = encode_applicant(PRESETS["experienced_manager"])
    response = predict.app.test_client().post("/predict", json=payload)
    assert response.status_code == 200
    body = response.get_json()
    assert body["income_class"] == 1.0
    assert "greater than $50,000" in body["message"]
    assert body["probability_over_50k"] > 0.5


def test_predict_coerces_numeric_strings():
    """Older clients sent age as a string. That used to be scored as zero."""
    payload = encode_applicant(PRESETS["part_time"])
    payload["age"] = "22"
    response = predict.app.test_client().post("/predict", json=payload)
    assert response.status_code == 200
    assert response.get_json()["income_class"] == 0.0


def test_predict_rejects_a_list_body():
    response = predict.app.test_client().post("/predict", json=[])
    assert response.status_code == 400


def test_recent_predictions_when_mongo_is_down():
    response = predict.app.test_client().get("/predictions")
    assert response.status_code == 503
    assert "unavailable" in response.get_json()["error"]
