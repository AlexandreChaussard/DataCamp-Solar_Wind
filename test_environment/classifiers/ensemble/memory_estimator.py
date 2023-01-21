import sklearn.preprocessing

from test_environment.utils import get_preprocessing, FeatureExtractor, Pipeline
from test_environment.feature_extractor import FeatureExtractor_ElasticMemory
from test_environment.classifiers.ensemble.estimator import EnsembleClassifier

from sklearn.pipeline import make_pipeline
import sklearn.preprocessing as preprocessing

from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin
import numpy as np
import pandas as pd


class MemoryEnsembleClassifier(MultiOutputMixin, ClassifierMixin, BaseEstimator):
    """
    This classifier is an ensemble learning ecosystem made of different XGBoost that have different weighting on
    each classes. So hopefully their differences make the model more accurate in the end.
    """

    def __init__(
            self,
            short_memory_extractor,
            medium_memory_extractor,
            long_memory_extractor,
            moving_avg=6,
            smoothing_threshold=0.5,
            random_state=None
    ):
        self.random_state = random_state

        self.short_memory_pipe = short_memory_extractor
        self.medium_memory_pipe = medium_memory_extractor
        self.long_memory_pipe = long_memory_extractor

        self.short_memory_model = EnsembleClassifier(moving_avg=4, smoothing_threshold=0.5)
        self.medium_memory_model = EnsembleClassifier(moving_avg=5, smoothing_threshold=0.4)
        self.long_memory_model = EnsembleClassifier(moving_avg=10, smoothing_threshold=0.3)

        self.short_memory_standardizer = preprocessing.StandardScaler()
        self.medium_memory_standardizer = preprocessing.StandardScaler()
        self.long_memory_standardizer = preprocessing.StandardScaler()
        self.short_memory_scaler = preprocessing.MinMaxScaler()
        self.medium_memory_scaler = preprocessing.MinMaxScaler()
        self.long_memory_scaler = preprocessing.MinMaxScaler()


        self.moving_avg = moving_avg
        self.smoothing_threshold = smoothing_threshold

    def fit(self, X, y, sample_weight=None):

        X_short_memory = self.short_memory_pipe.transform(X)
        X_medium_memory = self.medium_memory_pipe.transform(X)
        X_long_memory = self.long_memory_pipe.transform(X)

        X_short_memory = self.short_memory_standardizer.fit_transform(X_short_memory, y)
        X_medium_memory = self.medium_memory_standardizer.fit_transform(X_medium_memory, y)
        X_long_memory = self.long_memory_standardizer.fit_transform(X_long_memory, y)
        X_short_memory = self.short_memory_scaler.fit_transform(X_short_memory, y)
        X_medium_memory = self.medium_memory_scaler.fit_transform(X_medium_memory, y)
        X_long_memory = self.long_memory_scaler.fit_transform(X_long_memory, y)

        print("[*] Fitting memory estimator...")

        self.short_memory_model = self.short_memory_model.fit(X_short_memory, y)
        self.medium_memory_model = self.medium_memory_model.fit(X_medium_memory, y)
        self.long_memory_model = self.long_memory_model.fit(X_long_memory, y)

        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        X_short_memory = self.short_memory_pipe.transform(X)
        X_medium_memory = self.medium_memory_pipe.transform(X)
        X_long_memory = self.long_memory_pipe.transform(X)

        X_short_memory = self.short_memory_standardizer.transform(X_short_memory)
        X_medium_memory = self.medium_memory_standardizer.transform(X_medium_memory)
        X_long_memory = self.long_memory_standardizer.transform(X_long_memory)
        X_short_memory = self.short_memory_scaler.transform(X_short_memory)
        X_medium_memory = self.medium_memory_scaler.transform(X_medium_memory)
        X_long_memory = self.long_memory_scaler.transform(X_long_memory)

        probas_short_memory = self.short_memory_model.predict_proba(X_short_memory)
        probas_medium_memory = self.medium_memory_model.predict_proba(X_medium_memory)
        probas_long_memory = self.long_memory_model.predict_proba(X_long_memory)

        predictions = [probas_short_memory, probas_medium_memory, probas_long_memory]

        n_classes = 2
        probas = np.zeros((X.shape[0], n_classes))

        for prediction in predictions:
            for c in range(n_classes):
                probas[:, c] += prediction[:, c] / len(predictions)

        probas = probas / probas.sum(axis=1)[:, np.newaxis]

        return probas

    def predict(self, X):
        predictions = np.argmax(self.predict_proba(X), axis=1)
        predictions = pd.DataFrame(data=predictions).rolling(self.moving_avg).mean().ffill().bfill().values
        predictions[predictions > self.smoothing_threshold] = 1
        predictions[predictions <= self.smoothing_threshold] = 0
        return predictions


def get_estimator() -> Pipeline:
    feature_extractor_short_memory = FeatureExtractor_ElasticMemory(
        memories=['1h', '3h', '5h'],
        timelags=[1, 5, 10, 20, 40,  -1, -5, -10, -20, -40]
    )

    feature_extractor_medium_memory = FeatureExtractor_ElasticMemory(
        memories=['10h', '20h', '30h', '50h', '80h'],
        timelags=[60, 120, 300, -60, -120, -300]
    )

    feature_extractor_long_memory = FeatureExtractor_ElasticMemory(
        memories=['80h', '90h', '100h'],
        timelags=[480, 600, -480, -600]
    )

    pipe = make_pipeline(
        MemoryEnsembleClassifier(
            short_memory_extractor=feature_extractor_short_memory,
            medium_memory_extractor=feature_extractor_medium_memory,
            long_memory_extractor=feature_extractor_long_memory,
            moving_avg=2,
            smoothing_threshold=0.4)
    )

    return pipe
