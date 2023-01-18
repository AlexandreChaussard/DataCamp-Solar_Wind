from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.pipeline import make_pipeline

from test_environment.utils import get_preprocessing, FeatureExtractor, Pipeline


def get_estimator() -> Pipeline:
    feature_extractor = FeatureExtractor()

    model = HistGradientBoostingClassifier(max_iter=300,
                                           loss='log_loss',
                                           max_depth=3,
                                           l2_regularization=10,
                                           early_stopping=True,
                                           validation_fraction=0.5,
                                           tol=10e-3,
                                           class_weight={0: 1, 1: 1.3})
    classifier = BaggingClassifier(
        estimator=model,
        n_estimators=50,
        max_samples=1.0,
        max_features=1.0,
        oob_score=True,
        bootstrap=True,
        n_jobs=-1
    )

    pipe = make_pipeline(
        feature_extractor,
        *get_preprocessing(),
        classifier
    )
    return pipe
