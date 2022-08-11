import argparse
import os
import pickle

import mlflow
from hyperopt import hp, space_eval
from hyperopt.pyll import scope

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from xgboost import XGBClassifier
from sklearn.metrics import f1_score

HPO_EXPERIMENT_NAME = 'xgboost-hyperopt'
EXPERIMENT_NAME = 'xgboost-classifiers'

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment(EXPERIMENT_NAME)

# mlflow.sklearn.autolog()

SPACE = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 10, 0.1)),
    'learning_rate': hp.quniform('learning_rate', 0.01, 0.5, 0.01),
    'n_estimators': hp.choice('n_estimators', range(0, 50, 1)),
    'gamma': hp.quniform('gamma', 0.01, 0.50, 0.01),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 0.1),
    'subsample': hp.quniform('subsample', 0.1, 1, 0.01),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1.0, 0.01),
    'seed': 42
}


def load_pickle(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def train_and_log_model(datapath, params):
    X_train, y_train = load_pickle(os.path.join(datapath, 'train.pkl'))
    X_valid, y_valid = load_pickle(os.path.join(datapath, 'val.pkl'))
    X_test, y_test = load_pickle(os.path.join(datapath, 'test.pkl'))

    with mlflow.start_run():
        params = space_eval(SPACE, params)
        print(f"params: {params}")
        clf = XGBClassifier(**params)
        clf.fit(X_train, y_train)

        # Evaluate the model on the validation and test set
        valid_f1 = f1_score(y_valid, clf.predict(X_valid))
        mflow.log_metric('valid_f1', valid_f1)
        print(f"valid_f1: {valid_f1}")
        test_f1 = f1_score(y_test, clf.predict(X_test))
        mlflow.log_metric('test_f1', test_f1)
        print(f"test_f1: {test_f1}")


def run(datapath, logged_models):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models to MLFlow
    # client.create_experiment(HPO_EXPERIMENT_NAME)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=logged_models,
        order_by=['metrics.test_f1 DESC']
    )
    for run in runs:
        train_and_log_model(datapath=datapath, params=run.data.params)

    # Select the model with the highest test_f1 metric

    best_run = runs[0].info.run_id
    print(f"best_run: {best_run}")

    model_uri = f"runs:/{best_run}/models_mlflow"

    # register the best model
    mlflow.register_model(
        model_uri=model_uri,
        name='xgb-classifier'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datapath',
        type=str,
        default='../data/processed/',
        help='path to processed data'
    )

    parser.add_argument(
        "--top_n",
        default=10,
        type=int,
        help="the top n models to log and evaluate"
    )
    args = parser.parse_args()

    run(args.datapath, args.top_n)
    print('Done')
