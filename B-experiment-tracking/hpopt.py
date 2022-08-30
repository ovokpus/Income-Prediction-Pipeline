import argparse
import os
import pickle

import mlflow
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("xgboost-experiment")


def load_pickle(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def run(datapath, num_trials):

    X_train, y_train = load_pickle(os.path.join(datapath, 'train.pkl'))
    X_valid, y_valid = load_pickle(os.path.join(datapath, 'val.pkl'))

    def objective(params):
        with mlflow.start_run():
            mlflow.log_params(params)
            clf = XGBClassifier(**params)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_valid)

            f1 = f1_score(y_valid, y_pred)
            auc = roc_auc_score(y_valid, y_pred)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("auc", auc)
            print("f1 score: {}".format(f1))
            print("auc score: {}".format(auc))

            # mlflow.xgboost.log_model(clf, artifact_path="models_mlflow")

            # mlflow.log_artifact(os.path.join(
            #     datapath, 'train.pkl'), artifact_path='models_mlflow')

        return {'loss': -f1, 'status': STATUS_OK}

    space = {
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

    fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=num_trials,
        rstate=rstate
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--datapath',
        type=str,
        default='./data/processed'
    )

    parser.add_argument(
        '--num_trials',
        type=int,
        default=400,
        help='the number of evaluations to run/explore'
    )

    args = parser.parse_args()
    run(args.datapath, args.num_trials)
    print("done")
