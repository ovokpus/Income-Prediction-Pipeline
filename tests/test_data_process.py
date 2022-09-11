"""
Unit tests for data processing
"""
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from B_experimentTracking import preprocess

test_features = pd.read_csv(
    "/home/ovokpus/Income-Prediction-Pipeline/tests/df_new_test.csv"
)
test_target = pd.read_csv(
    "/home/ovokpus/Income-Prediction-Pipeline/tests/y_train_test.csv"
)
ACTUAL_DATA_PATH = "/home/ovokpus/Income-Prediction-Pipeline/data/adult-train.csv"


def test_read_data():
    """Tests the function that reads data from input directory

    Keyword arguments:
    argument -- description
    Return: return_description
    """

    expected_x = test_features
    expected_y = test_target

    actual_x, actual_y = preprocess.read_data(ACTUAL_DATA_PATH)

    assert actual_x.columns.all() == expected_x.columns.all()
    assert actual_y.name == "incomeTarget"
    assert expected_y.columns.to_list()[0] == actual_y.name


def test_preprocess_data():
    """Tests data preprocessing code

    Keyword arguments:
    argument -- description
    Return: return_description
    """

    dicts = test_features.to_dict(orient="records")
    dict_vectorizer = DictVectorizer()
    expected_df = dict_vectorizer.fit_transform(dicts)
    actual_x, _ = preprocess.read_data(ACTUAL_DATA_PATH)
    actual_df, _ = preprocess.preprocess_data(actual_x, dict_vectorizer, True)
    assert expected_df.shape[1] == actual_df.shape[1]
