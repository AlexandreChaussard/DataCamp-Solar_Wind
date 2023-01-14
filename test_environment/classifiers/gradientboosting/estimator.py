from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

from test_environment.utils import get_preprocessing, FeatureExtractor, Pipeline


def get_estimator() -> Pipeline:
    feature_extractor = FeatureExtractor()

    classifier = HistGradientBoostingClassifier(max_iter=50,
                                                max_depth=10,
                                                l2_regularization=0.02,
                                                class_weight={0: 0.6, 1: 0.4})

    pipe = make_pipeline(
        feature_extractor,
        *get_preprocessing(),
        classifier
    )

    return pipe
