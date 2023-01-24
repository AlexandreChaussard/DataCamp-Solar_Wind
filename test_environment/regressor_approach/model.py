from sklearn.pipeline import make_pipeline, Pipeline
import numpy as np
import pandas as pd

from test_environment.feature_extractor import FeatureExtractor_ElasticMemory, get_preprocessing, Pipeline, sliding_label
from test_environment.classifiers.ensemble.estimator import EnsembleClassifier

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier


def get_estimator() -> Pipeline:
    feature_extractor = FeatureExtractor_ElasticMemory(
        memories=["1h", "2h", "3h", "5h",
                  "10h", "15h", "20h", "30h", "50h", "70h",
                  "80h", "100h"],
        timelags=[6, 12, 18, 30,
                  60, 90, 120, 180, 300, 420,
                  480, 600]
    )

    regressor = HistGradientBoostingRegressor(max_iter=200,
                                              max_depth=4,
                                              learning_rate=10e-2,
                                              l2_regularization=1,
                                              validation_fraction=0.3,
                                              tol=10e-3)

    classifier = HistGradientBoostingClassifier(max_iter=200,
                                                max_depth=3,
                                                learning_rate=10e-2,
                                                l2_regularization=1,
                                                early_stopping=True,
                                                validation_fraction=0.3,
                                                tol=10e-3,
                                                scoring='accuracy',
                                                class_weight={0: 1, 1: 1})

    regressor_classifier = RegressorToClassifier(regressor=regressor,
                                                 classifier=classifier,
                                                 sliding_window=120,
                                                 moving_avg=4,
                                                 smoothing_threshold=0.57)

    pipe = make_pipeline(
        feature_extractor,
        *get_preprocessing(),
        regressor_classifier
    )

    return pipe


class RegressorToClassifier:

    def __init__(self, regressor, classifier, sliding_window, moving_avg, smoothing_threshold):
        self.regressor = regressor
        self.classifier = classifier
        self.sliding_window = sliding_window

        # Smoothing parameters
        self.moving_avg = moving_avg
        self.smoothing_threshold = smoothing_threshold

    def fit(self, X, y, sample_weight=None):

        # We turn the y into a continous flow containing the time series aspect of the problem
        print("Sliding window")
        y_reg = sliding_label(y.values, self.sliding_window)
        # We fit the regressor
        print("Fitting regressor")
        self.regressor.fit(X, y_reg)
        # We compute the theoritical output corresponding to it
        print("Predicting regressor")
        y_pred_reg = self.regressor.predict(X).reshape(-1, 1)
        X_reg = np.concatenate((X, y_pred_reg), axis=1)
        # We fit the classifier on the regression output to predict the actual class
        print("Fitting classifier")
        self.classifier.fit(X_reg, y)

        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        print("Predicting regressor proba")
        y_pred_reg = self.regressor.predict(X).reshape(-1, 1)
        X_reg = np.concatenate((X, y_pred_reg), axis=1)
        print("Prediciting classifier proba")
        return self.classifier.predict_proba(X_reg)

    def predict(self, X):
        predictions = self.predict_proba(X)[:, 1]
        predictions = pd.DataFrame(data=predictions).rolling(self.moving_avg).mean().ffill().bfill().values
        predictions[predictions > self.smoothing_threshold] = 1
        predictions[predictions <= self.smoothing_threshold] = 0
        return predictions
