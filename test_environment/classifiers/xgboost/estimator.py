from sklearn.pipeline import make_pipeline
from test_environment.utils import get_preprocessing, FeatureExtractor, Pipeline

from sklearn.ensemble import GradientBoostingClassifier

def get_estimator() -> Pipeline:
    feature_extractor = FeatureExtractor()

    classifier = GradientBoostingClassifier(
        loss="log_loss",
        n_estimators=100,
        learning_rate=0.01,
        max_depth=3,
        max_features=30,
        validation_fraction=0.4,
        tol=10e-3
    )

    pipe = make_pipeline(
        feature_extractor,
        *get_preprocessing(),
        classifier
    )

    return pipe
