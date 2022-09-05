import os
import pickle
import requests
import pandas as pd
import xgboost

import mlflow
from flask import Flask, request, jsonify

from pymongo import MongoClient


mlflow.set_tracking_uri('http://10.138.0.5:5000')
RUN_ID = os.getenv("MODEL_RUN_ID")
logged_model = f"runs:/{RUN_ID}/models_mlflow"
model = mlflow.pyfunc.load_model(logged_model)

EVIDENTLY_SERVICE_ADDRESS = os.getenv(
    "EVIDENTLY_SERVICE", "http://0.0.0.0:8085")
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://0.0.0.0:27017")

with open("./models/preprocessor.b", "rb") as f1:
    dv = pickle.load(f1)

# with open("./models/clf.bin", "rb") as f:
#     model = pickle.load(f)


def predict(data):
    dict_features = dv.transform(data)
    preds = model.predict(dict_features)
    return float(preds[0])


app = Flask("income-prediction")
mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction-service")
collection = db.get_collection("data")


@app.route("/predict", methods=["POST"])
def predict_income():
    data_point = request.get_json()
    preds = predict(data_point)

    result = {
        'income_class': preds
        # 'model_version': RUN_ID
    }

    if result['income_class'] == 0:
        result['message'] = 'Individual has annual Income less than $50,000'
    else:
        result['message'] = 'Individual has annual Income greater than $50,000'

    save_to_db(data_point, preds)
    save_to_evidently_service(data_point, preds)

    return jsonify(result)


def save_to_db(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    collection.insert_one(rec)


def save_to_evidently_service(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/census", json=[rec])


if __name__ == "__main__":
    app.run(debug=True, host="10.138.0.5", port=9696)
