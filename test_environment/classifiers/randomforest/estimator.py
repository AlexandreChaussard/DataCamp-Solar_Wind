from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

from test_environment.utils import get_preprocessing, FeatureExtractor, Pipeline


def get_estimator() -> Pipeline:
    feature_extractor = FeatureExtractor()

    classifier = RandomForestClassifier(n_estimators=80,
                                        criterion='entropy',
                                        max_features=8,
                                        random_state=1,
                                        class_weight={0: 1, 1: 1.2})

    pipe = make_pipeline(
        feature_extractor,
        *get_preprocessing(),
        classifier
    )
    return pipe
