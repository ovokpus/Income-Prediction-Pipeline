import os
import pickle

import mlflow
from flask import Flask, request, jsonify

RUN_ID = os.getenv(key="MLFLOW_RUN_ID", default=None)

logged_model = f"runs:/{RUN_ID}/model"
model = mlflow.pyfunc.load_model(logged_model)


def prepare_data(data):
    features = {}
    return data
