import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

import mlflow

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from prefect import Flow, task
from prefect.task_runners import SequentialTaskRunner


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
def scale_data(df: pd.DataFrame, scaler: StandardScaler, fit_scaler: bool = False):
    if fit_scaler:
        X = scaler.fit_transform(df)
    X = scaler.transform(df)
    return pd.DataFrame(X, columns=df.columns)


@task
def preprocess_data(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    dicts = df.to_dict(orient='records')

    if fit_dv:
        df = dv.fit_transform(dicts)
    df = dv.transform(dicts)

    return df, dv


@task
def train_model_search(X_train, y_train, X_val, y_val):
    def _objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "XGBClassifier")
            mlflow.log_params(params)
            clf = XGBClassifier(**params)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            f1 = f1_score(y_val, y_pred)
            mlflow.log_metric("f1", f1)

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
        fn=objective,
        space=space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=num_trials,
        rstate=rstate
    )

    return best_result


@task
def get_top_model_params(top_n):
    '''
    get experiment by name, search runs, and get the best run
    '''
    return best_run_params


@task
def train_best_model(X_train, y_train, X_val, y_val, dv, best_params):
    '''
    train the best model
    '''
    with mlflow.start_run():
        best_params = get_top_model_params(top_n)

    return clf


@flow
def main_flow(trainpath: str = "../data/adult-train.csv",
              valpath: str = "../data/adult-val.csv"):
    """
    Main flow for the experiment.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("xgboost-experiment")

    # load the csv files
    X_train, y_train = read_data(trainpath)
    X_val, y_val = read_data(valpath)

    # scale the data
    scaler = StandardScaler()
    X_train_scaled = scale_data(X_train, scaler, fit_scaler=True)
    X_val_scaled = scale_data(X_val, scaler)

    # preprocess the data
    dv = DictVectorizer()
    X_train_preprocessed, dv = preprocess_data(X_train_scaled, dv, fit_dv=True)
    X_val_preprocessed = preprocess_data(X_val_scaled, dv)[0]

    # train the model
    # model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
    # model.fit(X_train_preprocessed, y_train)
    # y_val_pred = model.predict(X_val_preprocessed)
    # y_test_pred = model.predict(X_test_preprocessed)
    # print(f"Validation F1 score: {f1_score(y_val, y_val_pred)}")
    # print(f"Test F1 score: {f1_score(y_test, y_test_pred)}")
    # return model, dv
