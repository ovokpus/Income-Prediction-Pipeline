import argparse
import os
import pickle

from xgboost import XGBClassifier
from sklearn.metrics import f1_score


def load_pickle(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def run(datapath):
    X_train, y_train = load_pickle(os.path.join(datapath, 'train.pkl'))
    X_valid, y_valid = load_pickle(os.path.join(datapath, 'val.pkl'))

    clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)

    f1 = f1_score(y_valid, y_pred)
    print("f1 score: {}".format(f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datapath',
        type=str,
        default='../data/processed/',
        help='path to processed data'
    )
    args = parser.parse_args()

    run(args.datapath)
    print('Done')
