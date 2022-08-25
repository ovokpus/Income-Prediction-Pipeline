import os
import argparse

import uuid
import pickle

from datetime import datetime

import pandas as pd

import mlflow

from prefect import task, flow, get_run_logger
from prefect.context import get_run_context

from dateutil.relativedelta import relativedelta

from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier
from sklearnn.metrics import f1_score
from sklearn.pipeline import make_pipeline


def generate_uuids(n):
    record_ids = []
    for item in range(n):
        record_ids.append(str(uuid.uuid4()))
    return record_ids


def read_data(filepath):

    df = pd.read_csv(filepath)

    df.drop(['nativeCountry'], axis=1, inplace=True)

    target = 'incomeTarget'

    transformed_target = []

    for _, value in df['incomeTarget'].iteritems():
        if value == ' <=50K':
            transformed_target.append(0)
        else:
            transformed_target.append(1)
    df['incomeTarget'] = transformed_target

    y = df[target]
    X = df.drop('incomeTarget', axis=1, inplace=True)
    X = pd.get_dummies(df)

    # Upsample using SMOTE
    sm = SMOTE(random_state=12)
    X_train_sm, y_train_sm = sm.fit_resample(X, y)

    df_new = pd.DataFrame(X_train_sm, columns=X.columns)

    return df_new, y_train_sm


def scale_data(df: pd.DataFrame, scaler: StandardScaler, fit_scaler: bool = False):
    if fit_scaler:
        X = scaler.fit_transform(df)
    X = scaler.transform(df)
    return pd.DataFrame(X, columns=df.columns)


def preprocess_data(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    dicts = df.to_dict(orient='records')

    if fit_dv:
        df = dv.fit_transform(dicts)
    df = dv.transform(dicts)

    return df, dv


def load_model(run_id):
    # logged_model = mlflow.get_artifact_uri(run_id, 'model')
    with open("../models/clf.bin", "rb") as f:
        model = pickle.load(f)
    return model


def save_results(df, y_pred, run_id, output_file):
    df_results = pd.DataFrame()
    df_results['record_id'] = generate_uuids(len(y_pred))
    df_results['age'] = df['age']
    df_results['workClass'] = df['workClass']
    df_results['financialWeight'] = df['financialWeight']
    df_results['education'] = df['education']
    df_results['maritalStatus'] = df['maritalStatus']
    df_results['occupation'] = df['occupation']
    df_results['relationship'] = df['relationship']
    df_results['actualLabel'] = y_pred
    df_results['predictedLabel'] = df['incomeTarget']
    df_results['difference'] = df_results['actualLabel'] - \
        df_results['predictedLabel']
    df_results['model_version'] = run_id

    df_results.to_csv(output_file, index=False)


@task
def apply_model(input_file, run_id, output_file):
    logger = get_run_logger()
    logger.info(f"input_file: {input_file}")
    logger.info(f"run_id: {run_id}")
    logger.info(f"output_file: {output_file}")

    logger.info(f"reading data")
    df, y_train_sm = read_data(input_file)
    scaler = StandardScaler()
    scaled_df = scale_data(df, scaler)
    dv = DictVectorizer()
    vectorized_df, dv = preprocess_data(df, dv)

    logger.info(f"loading model")
    model = load_model(run_id)
    y_pred = model.predict(df)

    logger.info(f"saving results")
    save_results(df, y_pred, run_id, output_file)
    return output_file


def get_paths(input_file, output_file):
    input_file = os.path.join(os.getcwd(), input_file)
    output_file = os.path.join(os.getcwd(), output_file)
    return input_file, output_file


@flow
def income_prediction(input_file, output_file, run_id, run_date: datetime = None):
    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time

    input_file, output_file = get_paths(input_file, output_file)

    apply_model(input_file, run_id, output_file)


def run():
    parser = argparse.ArgumentParser(
        description='Run the income prediction flow')
    parser.add_argument('--input_file', type=str,
                        default='../data/adult-test.csv', help='input file')
    parser.add_argument('--output_file', type=str,
                        default='../data_output/income_prediction.csv', help='output file')
    parser.add_argument('--run_id', type=str, default=None, help='run id')
    parser.add_argument('--run_date', type=str, default=None, help='run date')
    args = parser.parse_args()
    income_prediction(args.input_file, args.output_file,
                      args.run_id, args.run_date)


if __name__ == '__main__':
    run()
