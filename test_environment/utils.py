import pandas as pd

import problem

from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.preprocessing as preprocessing

from sklearn.base import BaseEstimator

import numpy as np


def fetch_data(percentage=1.0):
    print(f"[Data] Fetching {int(percentage * 100)}% of the data...")
    data_train, label_train = problem.get_train_data(path="../")
    data_test, label_test = problem.get_test_data(path="../")

    index_train = int(len(data_train.index) * percentage)
    index_test = int(len(data_test.index) * percentage)
    data_train, label_train = data_train[:index_train], label_train[:index_train]
    data_test, label_test = data_test[:index_test], label_test[:index_test]

    return data_train, label_train, data_test, label_test


def run_estimator(
        estimator: Pipeline,
        name: str,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None
):
    print(f"[Pipeline] Running pipepline: {name}")
    # Fitting estimator
    print("           * Fitting the training set")
    estimator.fit(X_train, y_train)
    # Making prediction
    print("           * Predicting on the test set")
    y_pred = estimator.predict(X_test)
    # Print model information
    print("           * Printing report")
    report = classification_report(y_test, y_pred)
    print(report)
    print(f"           * Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred)}")
    # Printing confusion matrix
    print("           * Printing confusion matrix")
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    print(cm)


class FeatureExtractor(BaseEstimator):

    def __init__(self):
        super().__init__()
        self.n_features = 33 + 11

    def fit(self, X, y):
        return self

    def fill_missing_values(self, X_df: pd.DataFrame, new_feature, original_feature):
        X_df[new_feature] = X_df[new_feature].ffill().bfill()
        X_df[new_feature] = X_df[new_feature].fillna(method='ffill')
        X_df[new_feature] = X_df[new_feature].astype(X_df[original_feature].dtype)
        return X_df

    def compute_feature_lag(self, X_df, feature, time_shift):
        name = "_".join([feature, f"{time_shift}"])

        X_df[name] = X_df[feature]

        if time_shift < 0:
            missing_values = np.full(abs(time_shift), X_df[feature].iloc[0])
            shifted_feature = X_df[feature].iloc[0:time_shift].to_numpy()
            shifted = np.concatenate((missing_values, shifted_feature))
        else:
            missing_values = np.full(time_shift, X_df[feature].iloc[-1])
            shifted_feature = X_df[feature].iloc[time_shift:].to_numpy()
            shifted = np.concatenate((shifted_feature, missing_values))

        X_df.loc[:, name] = shifted
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def compute_rolling_max(self, X_df: pd.DataFrame, feature, time_window, center=True):
        name = "_".join([feature, time_window, "max"])
        X_df[name] = X_df[feature].rolling(time_window, center=center).max()
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def compute_rolling_min(self, X_df: pd.DataFrame, feature, time_window, center=True):
        name = "_".join([feature, time_window, "min"])
        X_df[name] = X_df[feature].rolling(time_window, center=center).min()
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def compute_rolling_mean(self, X_df: pd.DataFrame, feature, time_window, center=True):
        name = "_".join([feature, time_window, "mean"])
        X_df[name] = X_df[feature].rolling(time_window, center=center).mean()
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def compute_rolling_var(self, X_df: pd.DataFrame, feature, time_window, center=True):
        name = "_".join([feature, time_window, "var"])
        X_df[name] = X_df[feature].rolling(time_window, center=center).var()
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def transform(self, X):
        X_df = self.compute_rolling_var(X, "Beta", "2h")
        X_df = self.compute_rolling_var(X_df, "Beta", "20min")
        X_df = self.compute_rolling_min(X_df, "Beta", "2h")
        X_df = self.compute_rolling_min(X_df, "Beta", "30min")
        X_df = self.compute_rolling_max(X_df, "Beta", "1h")
        X_df = self.compute_rolling_max(X_df, "Beta", "30min")

        for feature in ["Beta_2h_min", "Beta", "RmsBob", "Vx", "Range F 9"]:
            X_df = self.compute_feature_lag(X_df, feature, -1)
            X_df = self.compute_feature_lag(X_df, feature, -5)
            X_df = self.compute_feature_lag(X_df, feature, -10)
            X_df = self.compute_feature_lag(X_df, feature, -20)
            X_df = self.compute_feature_lag(X_df, feature, 1)
            X_df = self.compute_feature_lag(X_df, feature, 5)
            X_df = self.compute_feature_lag(X_df, feature, 10)
            X_df = self.compute_feature_lag(X_df, feature, 20)
        return X_df


def get_preprocessing():
    return preprocessing.QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=1), \
           preprocessing.StandardScaler(), \
           preprocessing.MinMaxScaler()
