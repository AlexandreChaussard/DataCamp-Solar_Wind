from sklearn.pipeline import make_pipeline
from test_environment.utils import get_preprocessing, FeatureExtractor, Pipeline

from sklearn.neural_network import MLPClassifier


def get_estimator() -> Pipeline:
    feature_extractor = FeatureExtractor()

    classifier = MLPClassifier(
        hidden_layer_sizes=[100, 150, 150, 150, 150, 100],
        activation='relu',
        solver='adam',
        alpha=0.2,  # L2 regularizer
        learning_rate='constant',
        max_iter=2000,
        shuffle=True,
        tol=1e-4,  # plateau for the score
        momentum=0.9,
        validation_fraction=0.4,
    )

    pipe = make_pipeline(
        feature_extractor,
        *get_preprocessing(),
        classifier
    )
    return pipe
