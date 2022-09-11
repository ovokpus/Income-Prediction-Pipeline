import argparse
import os
import pickle

import mlflow
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

mlflow.set_tracking_uri("http://10.138.0.5:5000")
mlflow.set_experiment("xgboost-tracking")


def load_pickle(filename: str):
    with open(filename, "rb") as f:
        return pickle.load(f)


def run(datapath):
    with mlflow.start_run():

        X_train, y_train = load_pickle(os.path.join(datapath, "train.pkl"))
        X_valid, y_valid = load_pickle(os.path.join(datapath, "val.pkl"))

        clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_valid)

        f1 = f1_score(y_valid, y_pred)
        auc = roc_auc_score(y_valid, y_pred)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("auc", auc)
        print("f1 score: {}".format(f1))
        print("auc score: {}".format(auc))

        mlflow.xgboost.log_model(clf, artifact_path="models_mlflow")

        mlflow.log_artifact(
            os.path.join(datapath, "train.pkl"), artifact_path="models_mlflow"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        type=str,
        default="./data/processed/",
        help="path to processed data",
    )
    args = parser.parse_args()

    run(args.datapath)
    print("Done")
