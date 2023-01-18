from sklearn.ensemble import HistGradientBoostingClassifier

from test_environment.classifiers.markov.markov_model import MemoryClassifier

from sklearn.pipeline import make_pipeline

from test_environment.utils import get_preprocessing, FeatureExtractor, Pipeline


def get_estimator(memory=12, characteristic_storm_duration=50, residual=0.0) -> Pipeline:
    feature_extractor = FeatureExtractor()

    classifier = HistGradientBoostingClassifier(max_iter=200,
                                                loss='log_loss',
                                                max_depth=2,
                                                learning_rate=10e-2,
                                                l2_regularization=1,
                                                early_stopping=True,
                                                validation_fraction=0.5,
                                                tol=10e-3,
                                                class_weight={0: 1, 1: 1.6})

    classifier = MemoryClassifier(
        memory=memory,
        classifier=classifier,
        characteristic_storm_duration=characteristic_storm_duration,
        residual=residual
    )

    pipe = make_pipeline(
        feature_extractor,
        *get_preprocessing(),
        classifier
    )
    return pipe
