from test_environment.utils import get_preprocessing, FeatureExtractor, Pipeline
from test_environment.feature_extractor import FeatureExtractor_ElasticMemory

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin
import numpy as np
import pandas as pd


class EnsembleClassifier(MultiOutputMixin, ClassifierMixin, BaseEstimator):
    """
    This classifier is an ensemble learning ecosystem made of different XGBoost that have different weighting on
    each classes. So hopefully their differences make the model more accurate in the end.
    """

    def __init__(self, moving_avg=6, smoothing_threshold=0.5, random_state=None):
        self.models = []
        self.random_state = random_state
        self.moving_avg = moving_avg
        self.smoothing_threshold = smoothing_threshold
        self.add_xgboost(
            weight_minority_class=1,
            max_depth=4,
            validation_fraction=0.1,
        )
        self.add_xgboost(
            weight_minority_class=2.1,
            max_depth=4,
            validation_fraction=0.4,
        )
        self.add_xgboost(
            weight_minority_class=2.8,
            max_depth=4,
            validation_fraction=0.1,
        )

    def add_xgboost(self, weight_minority_class=2.0, max_depth=2, learning_rate=10e-2, validation_fraction=0.5):
        classifier = HistGradientBoostingClassifier(max_iter=200,
                                                    loss='log_loss',
                                                    max_depth=max_depth,
                                                    learning_rate=learning_rate,
                                                    l2_regularization=2,
                                                    early_stopping=True,
                                                    validation_fraction=validation_fraction,
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
        predictions = np.argmax(self.predict_proba(X), axis=1)
        predictions = pd.DataFrame(data=predictions).rolling(self.moving_avg).mean().ffill().bfill().values
        predictions[predictions > self.smoothing_threshold] = 1
        predictions[predictions <= self.smoothing_threshold] = 0
        return predictions


def get_estimator() -> Pipeline:
    # feature_extractor = FeatureExtractor()
    feature_extractor = FeatureExtractor_ElasticMemory(
        memories=["1h", "3h", "5h", "12h", "22h", "33h", "50h", "70h", "100h"],
        timelags=[1, 5, 10, 20, 50, 100, 300, 800, 1500, -1, -5, -10, -20, -50, -100, -300, -800, -1500]
    )

    classifier = EnsembleClassifier(moving_avg=4, smoothing_threshold=0.4)

    pipe = make_pipeline(
        feature_extractor,
        *get_preprocessing(),
        classifier
    )
    return pipe
