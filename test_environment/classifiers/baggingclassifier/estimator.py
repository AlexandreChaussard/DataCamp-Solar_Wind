from sklearn.ensemble import HistGradientBoostingClassifier

from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE

from sklearn.pipeline import make_pipeline

from test_environment.utils import get_preprocessing, FeatureExtractor, Pipeline


def get_estimator() -> Pipeline:
    feature_extractor = FeatureExtractor()

    classifier = BalancedBaggingClassifier(
        estimator=HistGradientBoostingClassifier(max_depth=8, l2_regularization=0.02),
        n_estimators=50,
        bootstrap=True,
        sampler=SMOTE(sampling_strategy=.5, k_neighbors=5),
        random_state=1,
    )

    pipe = make_pipeline(
        feature_extractor,
        *get_preprocessing(),
        classifier
    )
    return pipe