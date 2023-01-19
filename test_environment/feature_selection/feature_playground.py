from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import HistGradientBoostingClassifier

min_features_to_select = 1  # Minimum number of features to consider
classifier = HistGradientBoostingClassifier(max_iter=200,
                                                loss='log_loss',
                                                max_depth=2,
                                                learning_rate=10e-2,
                                                l2_regularization=1,
                                                early_stopping=True,
                                                validation_fraction=0.5,
                                                tol=10e-3,
                                                class_weight={0: 1, 1: 1.6})
cv = TimeSeriesSplit(5)

rfecv = RFECV(
    estimator=classifier,
    step=1,
    cv=cv,
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
    n_jobs=2,
)

rfecv.fit(X, y)

print(f"Optimal number of features: {rfecv.n_features_}")