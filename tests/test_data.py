from ml_patient_classifier.data import load_dataframe


def test_load_dataframe():
    df = load_dataframe("data/raw/heart.csv")

    # dataset should not be empty
    assert len(df) > 0

    # expected number of columns (11 features + target)
    assert df.shape[1] == 12