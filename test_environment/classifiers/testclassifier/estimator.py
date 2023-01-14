from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from test_environment.markov_model import MemoryClassifier

from sklearn.pipeline import make_pipeline

from test_environment.utils import get_preprocessing, FeatureExtractor, Pipeline


def get_estimator() -> Pipeline:
    feature_extractor = FeatureExtractor()

    #classifier = GradientBoostingClassifier(n_estimators=10, max_depth=6, random_state=1)
    classifier = MemoryClassifier(
        memory=12,
        classifier=RandomForestClassifier(n_estimators=50, max_depth=7)
    )

    pipe = make_pipeline(
        feature_extractor,
        *get_preprocessing(),
        classifier
    )
    return pipe
