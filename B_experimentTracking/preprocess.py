import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction import DictVectorizer


def dump_pickle(data, filename):
    with open(filename, 'wb') as f:
        return pickle.dump(data, f)


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



def preprocess_data(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    dicts = df.to_dict(orient='records')

    if fit_dv:
        df = dv.fit_transform(dicts)
    df = dv.transform(dicts)

    return df, dv


def run(raw_data_path: str, dest_data_path: str):
    # load the csv files
    X_train, y_train = read_data(
        os.path.join(raw_data_path, 'adult-train.csv'))
    X_val, y_val = read_data(os.path.join(raw_data_path, 'adult-val.csv'))
    X_test, y_test = read_data(os.path.join(raw_data_path, 'adult-test.csv'))

    # preprocess the data
    dv = DictVectorizer()
    X_train_vectorized, dv = preprocess_data(
        X_train, dv, fit_dv=True)
    X_val_vectorized, _ = preprocess_data(X_val, dv)
    X_test_vectorized, _ = preprocess_data(X_test, dv)

    # create destination directory if not exists
    os.makedirs(dest_data_path, exist_ok=True)

    # dump the data
    dump_pickle(dv, os.path.join(dest_data_path, 'dv.pkl'))
    dump_pickle((X_train_vectorized, y_train),
                os.path.join(dest_data_path, 'train.pkl'))
    dump_pickle((X_val_vectorized, y_val),
                os.path.join(dest_data_path, 'val.pkl'))
    dump_pickle((X_test_vectorized, y_test),
                os.path.join(dest_data_path, 'test.pkl'))

    print('Done')
    print('Preprocessed data saved to {}'.format(dest_data_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data-path', type=str,
                        default='./data/', help='path to raw data')
    parser.add_argument('--dest-data-path', type=str,
                        default='./data/processed/', help='path to processed data')
    args = parser.parse_args()

    # run the preprocessing
    run(args.raw_data_path, args.dest_data_path)
    print('Preprocessing completed.')
