import numpy as np
import pandas as pd
import pickle

from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

from datetime import datetime as dt

EXPERIMENT_NAME = "xgboost-pipeline"
mlflow.set_tracking_uri("http://10.138.0.5:5000")
mlflow.set_experiment(EXPERIMENT_NAME)


@task
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

@task
def preprocess_data(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    dicts = df.to_dict(orient='records')

    if fit_dv:
        df = dv.fit_transform(dicts)
    df = dv.transform(dicts)

    return df, dv


@task
def train_model_search(X_train, y_train, X_val, y_val, num_trials: int=500):
    def _objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "XGBClassifier")
            mlflow.log_params(params)
            clf = XGBClassifier(**params)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            f1 = f1_score(y_val, y_pred)
            mlflow.log_metric("f1", f1)
            mlflow.xgboost.log_model(clf, artifact_path="models_mlflow")

        return {'loss': 1 - f1, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 10, 0.1)),
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.5, 0.01),
        'n_estimators': hp.choice('n_estimators', range(0, 50, 1)),
        'gamma': hp.quniform('gamma', 0.01, 0.50, 0.01),
        'min_child_weight': hp.quniform('min_child_weight', 0, 10, 0.1),
        'subsample': hp.quniform('subsample', 0.1, 1, 0.01),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1.0, 0.01),
        'random_state': 42
    }

    trials = Trials()
    rstate = np.random.default_rng(42)  # for reproducible results

    best_result = fmin(
        fn=_objective,
        space=search_space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=num_trials,
        rstate=rstate
    )

    return best_result


@task
def train_best_model(X_train, y_train, X_val, y_val, dv):
    '''
    train the best model
    '''
    with mlflow.start_run():
        best_params = params = {
            'colsample_bytree': 0.92,
            'gamma': 0.03,
            'learning_rate': 0.31,
            'max_depth': '7',
            'min_child_weight': 4.5,
            'n_estimators': 42,
            'random_state': 42,
            'subsample': 0.9500000000000001
        }
        mlflow.log_params(best_params)

        clf = XGBClassifier(**best_params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        train_time = dt.now()
        mlflow.log_metric("f1", f1)

        with open("./models/preprocessor.b", "wb") as f:
            pickle.dump(dv, f)

        with open("./models/clf.bin", "wb") as f1:
            pickle.dump(clf, f1)

        mlflow.log_artifact("./models/preprocessor.b",
                            artifact_path=f"preprocessors")
        mlflow.xgboost.log_model(clf, artifact_path="models_mlflow")
    
    return clf



@flow(task_runner=SequentialTaskRunner())
def main_flow(trainpath: str = "./data/adult-train.csv",
              valpath: str = "./data/adult-val.csv"):
    """
    Main flow for the experiment.
    """
    EXPERIMENT_NAME = "xgboost-pipeline"
    mlflow.set_tracking_uri('http://10.138.0.5:5000')
    mlflow.set_experiment(EXPERIMENT_NAME)

    # load the csv files
    X_train, y_train = read_data(trainpath).result()
    X_val, y_val = read_data(valpath).result()

    # preprocess the data
    dv = DictVectorizer()
    X_train_preprocessed, dv = preprocess_data(
        X_train, dv, fit_dv=True).result()
    X_val_preprocessed = preprocess_data(X_val, dv).result()[0]

    # train the model
    best_clf = train_model_search(X_train, y_train, X_val, y_val).result()
    train_best_model(X_train, y_train, X_val, y_val,
                     dv, wait_for=best_clf).result()
    
    
    client = MlflowClient()
     # Select the model with the highest test_f1 metric
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=['metrics.f1 DESC']
    )[0]

    print(f"best_run: {best_run}")

    model_uri = f"runs:/{best_run.info.run_id}/models_mlflow"
    print("model uri is: ", model_uri)

    # register the best model
    mlflow.register_model(
        model_uri=model_uri,
        name='xgb-classifier'
    )
    


from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

DeploymentSpec(
    flow=main_flow,
    name="income_prediction_training",
    schedule=IntervalSchedule(interval=timedelta(minutes=240)),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml", "xgboost"],
)


