import os
import pickle
import pandas as pd
import xgboost

import mlflow
from flask import Flask, request, jsonify


mlflow.set_tracking_uri('http://10.138.0.5:5000')
RUN_ID = os.getenv("MODEL_RUN_ID")
logged_model = f"runs:/{RUN_ID}/models_mlflow"
model = mlflow.pyfunc.load_model(logged_model)

with open("./models/preprocessor.b", "rb") as f1:
    dv, scaler = pickle.load(f1)

# with open("./models/clf.bin", "rb") as f:
#     model = pickle.load(f)


def predict(data):
    dict_features = dv.transform(data)
    scaled_features = scaler.transform(
        dict_features)
    preds = model.predict(scaled_features)
    return float(preds[0])


app = Flask("income-prediction")


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

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="10.138.0.5", port=8080)
