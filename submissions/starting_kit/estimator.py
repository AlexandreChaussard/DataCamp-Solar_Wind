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
        X_df = self.compute_rolling_var(X, "Beta", "1h")
        X_df = self.compute_rolling_var(X_df, "Beta", "20min")
        X_df = self.compute_rolling_var(X_df, "Np", "40min")
        X_df = self.compute_rolling_var(X_df, "Np_nl", "40min")

        X_df = self.compute_rolling_min(X_df, "By", "30min")
        X_df = self.compute_rolling_min(X_df, "Bz", "30min")
        X_df = self.compute_rolling_min(X_df, "By", "30min")
        X_df = self.compute_rolling_min(X_df, "Na_nl", "30min")
        X_df = self.compute_rolling_min(X_df, "Vx", "40min")
        X_df = self.compute_rolling_min(X_df, "Vy", "40min")

        X_df = self.compute_rolling_min(X_df, "By", "2h")
        X_df = self.compute_rolling_min(X_df, "Bz", "2h")
        X_df = self.compute_rolling_min(X_df, "By", "2h")
        X_df = self.compute_rolling_min(X_df, "Na_nl", "2h")
        X_df = self.compute_rolling_min(X_df, "Vx", "2h")
        X_df = self.compute_rolling_min(X_df, "Vy", "2h")

        X_df = self.compute_rolling_max(X_df, "Beta", "1h")
        X_df = self.compute_rolling_max(X_df, "Beta", "30min")
        X_df = self.compute_rolling_max(X_df, "B", "40min")
        X_df = self.compute_rolling_max(X_df, "Np", "40min")
        X_df = self.compute_rolling_max(X_df, "Np_nl", "40min")
        X_df = self.compute_rolling_max(X_df, "Range F 0", "40min")
        X_df = self.compute_rolling_max(X_df, "Range F 1", "40min")
        X_df = self.compute_rolling_max(X_df, "Range F 10", "40min")
        X_df = self.compute_rolling_max(X_df, "V", "40min")
        X_df = self.compute_rolling_max(X_df, "RmsBob", "40min")
        X_df = self.compute_rolling_max(X_df, "Beta", "2h")
        X_df = self.compute_rolling_max(X_df, "B", "2h")
        X_df = self.compute_rolling_max(X_df, "Np", "2h")
        X_df = self.compute_rolling_max(X_df, "Np_nl", "2h")
        X_df = self.compute_rolling_max(X_df, "Range F 0", "2h")
        X_df = self.compute_rolling_max(X_df, "Range F 1", "2h")
        X_df = self.compute_rolling_max(X_df, "Range F 10", "2h")
        X_df = self.compute_rolling_max(X_df, "V", "2h")
        X_df = self.compute_rolling_max(X_df, "RmsBob", "2h")

        for feature in ["Beta", "RmsBob", "Vx", "Range F 9", "Beta_30min_max"]:
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


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin


class EnsembleClassifier(MultiOutputMixin, ClassifierMixin, BaseEstimator):

    def __init__(self, random_state=None):
        self.models = []
        self.random_state = random_state
        self.add_xgboost(
            weight_minority_class=1,
            max_depth=4
        )
        self.add_xgboost(
            weight_minority_class=3,
            max_depth=2
        )

    def add_xgboost(self, weight_minority_class=1.6, max_depth=2, learning_rate=10e-2):
        classifier = HistGradientBoostingClassifier(max_iter=200,
                                                    loss='log_loss',
                                                    max_depth=max_depth,
                                                    learning_rate=learning_rate,
                                                    l2_regularization=1,
                                                    early_stopping=True,
                                                    validation_fraction=0.5,
                                                    tol=10e-3,
                                                    class_weight={0: 1, 1: weight_minority_class},
                                                    random_state=self.random_state)
        self.models.append(classifier)

    def fit(self, X, y, sample_weight=None):
        for model in self.models:
            model.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        probas = np.zeros((X.shape[0], len(self.classes_)))

        for model in self.models:
            predictions = model.predict_proba(X)
            for c in range(len(self.classes_)):
                probas[:, c] += predictions[:, c] / len(self.models)

        probas = probas / probas.sum(axis=1)[:, np.newaxis]
        return probas

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def get_estimator() -> Pipeline:
    feature_extractor = FeatureExtractor()

    classifier = EnsembleClassifier()

    pipe = make_pipeline(
        feature_extractor,
        *get_preprocessing(),
        classifier
    )

    return pipe
