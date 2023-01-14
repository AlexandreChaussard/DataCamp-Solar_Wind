# General imports
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator

from sklearn.pipeline import make_pipeline


# Model imports


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


from sklearn.ensemble import RandomForestClassifier


def get_estimator() -> Pipeline:
    feature_extractor = FeatureExtractor()

    classifier = RandomForestClassifier(n_estimators=50,
                                        max_depth=9,
                                        criterion='entropy',
                                        random_state=1,
                                        class_weight={0: 1, 1: 1.5})

    pipe = make_pipeline(
        feature_extractor,
        *get_preprocessing(),
        classifier
    )
    return pipe
