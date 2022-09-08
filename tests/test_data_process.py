import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from B_experimentTracking import preprocess

test_features = pd.read_csv('df_new_test.csv')
test_target = pd.read_csv('y_train_test.csv')
actual_data_path = '../data/adult-train.csv'


def test_read_data():
    expected_X = test_features
    expected_y = test_target

    actual_X, actual_y = preprocess.read_data(actual_data_path)

    assert actual_X.columns.all() == expected_X.columns.all()
    assert actual_y.name == 'incomeTarget'


def test_preprocess_data():
    dicts = test_features.to_dict(orient='records')
    dv = DictVectorizer()
    expected_df = dv.fit_transform(dicts)
    actual_X, actual_y = preprocess.read_data(actual_data_path)
    actual_df, _ = preprocess.preprocess_data(actual_X, dv, True)
    assert expected_df.shape[1] == actual_df.shape[1]
