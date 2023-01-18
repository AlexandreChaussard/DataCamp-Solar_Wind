from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

from test_environment.utils import get_preprocessing, FeatureExtractor, Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin
import numpy as np


class EnsembleClassifier(MultiOutputMixin, ClassifierMixin, BaseEstimator):

    def __init__(self, random_state=None):
        self.models = []
        self.random_state = random_state
        self.add_xgboost(
            weight_minority_class=1,
            max_depth=4,
            validation_fraction=0.1,
        )
        self.add_xgboost(
            weight_minority_class=3,
            max_depth=2,
            validation_fraction=0.5,
        )

    def add_xgboost(self, weight_minority_class=1.6, max_depth=2, learning_rate=10e-2, validation_fraction=0.5):
        classifier = HistGradientBoostingClassifier(max_iter=200,
                                                    loss='log_loss',
                                                    max_depth=max_depth,
                                                    learning_rate=learning_rate,
                                                    l2_regularization=1,
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