import pandas as pd


def load_data(path, label_col_name):
    X = pd.read_csv(path, index_col=0)
    y = X[label_col_name]
    X.drop(label_col_name, axis=1, inplace=True)
    return X, y
