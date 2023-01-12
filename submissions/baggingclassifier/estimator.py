import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline

from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from submissions.baggingclassifier.markov_random_forest import MarkovRandomForest

import sklearn.preprocessing as preprocessing


def compute_rolling_std(X_df, feature, time_window, center=False):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature
    Parameters
    ----------
    X_df : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling std from
    time_window : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    name = "_".join([feature, time_window, "std"])
    X_df[name] = X_df[feature].rolling(time_window, center=center).std()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df


class FeatureExtractor(BaseEstimator):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X_df = compute_rolling_std(X, "Beta", "2h")
        X_df = compute_rolling_std(X_df, "Beta", "15min")
        return X_df


def get_preprocessing():
    return preprocessing.QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=1), \
           preprocessing.StandardScaler(), \
           preprocessing.MinMaxScaler()


def get_estimator():
    feature_extractor = FeatureExtractor()

    classifier = BalancedBaggingClassifier(
        estimator=MarkovRandomForest(
            n_estimators=10,
            max_depth=4,
            criterion='gini',
            random_state=1),
        n_estimators=10,
        random_state=1
    )

    pipe = make_pipeline(
        feature_extractor,
        *get_preprocessing(),
        classifier
    )
    return pipe
