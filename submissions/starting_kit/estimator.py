# General imports
from sklearn import preprocessing
import sklearn.impute as impute
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator

from sklearn.pipeline import make_pipeline
from scipy import signal
from scipy import stats


# Model imports


class FeatureExtractor(BaseEstimator):

    def __init__(self):
        super().__init__()
        self.n_features = 33 + 11

    def fit(self, X, y):
        return self

    def smooth(self, X_df: pd.DataFrame, feature, time_window, center=False):
        X_df[feature] = X_df[feature].rolling(time_window, center=center).mean()
        return X_df

    def fill_missing_values(self, X_df: pd.DataFrame, new_feature, original_feature):
        X_df[new_feature] = X_df[new_feature].ffill().bfill()
        X_df[new_feature] = X_df[new_feature].fillna(method='ffill')
        X_df[new_feature] = X_df[new_feature].astype(X_df[original_feature].dtype)
        return X_df

    def compute_feature_lag(self, X_df, feature, time_shift):
        name = "_".join([feature, f"{time_shift}"])

        if time_shift < 0:
            missing_values = np.full(abs(time_shift), X_df[feature].iloc[0])
            shifted_feature = X_df[feature].iloc[0:time_shift].to_numpy()
            shifted = np.concatenate((missing_values, shifted_feature))
        else:
            missing_values = np.full(time_shift, X_df[feature].iloc[-1])
            shifted_feature = X_df[feature].iloc[time_shift:].to_numpy()
            shifted = np.concatenate((shifted_feature, missing_values))

        X_df[name] = shifted
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def compute_rolling_max(self, X_df: pd.DataFrame, feature, time_window, center=True):
        name = "_".join([feature, time_window, "max"])
        X_df[name] = X_df[feature].rolling(time_window, center=center).max()
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def compute_rolling_min(self, X_df: pd.DataFrame, feature, time_window, center=True):
        name = "_".join([feature, time_window, "min"])
        X_df[name] = X_df[feature].rolling(time_window, center=center).min()
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def compute_rolling_mean(self, X_df: pd.DataFrame, feature, time_window, center=True):
        name = "_".join([feature, time_window, "mean"])
        X_df[name] = X_df[feature].rolling(time_window, center=center).mean()
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def compute_rolling_median(self, X_df: pd.DataFrame, feature, time_window, center=True):
        name = "_".join([feature, time_window, "mean"])
        X_df[name] = X_df[feature].rolling(time_window, center=center).median()
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def compute_rolling_var(self, X_df: pd.DataFrame, feature, time_window, center=True):
        name = "_".join([feature, time_window, "var"])
        X_df[name] = X_df[feature].rolling(time_window, center=center).var()
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def compute_rolling_quantile(self, X_df: pd.DataFrame, feature, time_window, quantile):
        name = "_".join([feature, time_window, f"q{quantile}"])
        X_df[name] = X_df[feature].rolling(time_window, center=True).quantile(quantile, interpolation="linear")
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def compute_rolling_sum(self, X_df: pd.DataFrame, feature, time_window):
        name = "_".join([feature, time_window, f"sum"])
        X_df[name] = X_df[feature].rolling(time_window, center=True).sum()
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def compute_rolling_peaks_height(self, X_df: pd.DataFrame, feature, time_window):
        name = "_".join([feature, time_window, f"peaks_height"])
        number_peaks = np.zeros(X_df.shape[0])
        i = 0
        for window in X_df[feature].rolling(time_window, center=True):
            peaks, properties = signal.find_peaks(window.values, height=[0])
            peaks_height = properties["peak_heights"].sum()
            number_peaks[i] = peaks_height
            i += 1
        X_df[name] = number_peaks
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def compute_rolling_count_peaks(self, X_df: pd.DataFrame, feature, time_window):
        name = "_".join([feature, time_window, f"peaks_count"])
        number_peaks = np.zeros(X_df.shape[0])
        i = 0
        for window in X_df[feature].rolling(time_window, center=True):
            peaks, attributes = signal.find_peaks(window)
            number_peaks[i] = len(peaks)
            i += 1
        X_df[name] = number_peaks
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def compute_rolling_energy(self, X_df: pd.DataFrame, feature, time_window):
        name = "_".join([feature, time_window, f"energy"])
        squared_values = X_df[feature].apply(lambda x: x ** 2)
        X_df[name] = squared_values.rolling(time_window, center=True).sum()
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def compute_rolling_skewness(self, X_df: pd.DataFrame, feature, time_window):
        name = "_".join([feature, time_window, f"skew"])
        X_df[name] = X_df[feature].rolling(time_window, center=True).skew()
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def compute_rolling_entropy(self, X_df: pd.DataFrame, feature, time_window):
        name = "_".join([feature, time_window, f"entropy"])
        rolling_values = X_df[feature].rolling(time_window, center=True)
        entropy = np.zeros(X_df.shape[0])
        i = 0
        for value in rolling_values:
            entropy[i] = stats.entropy(value)
            i += 1
        X_df[name] = entropy
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def compute_cwt(self, X_df: pd.DataFrame, feature, width):
        name = "_".join([feature, "cwt", f"{width}"])
        values = signal.cwt(X_df[feature].values, signal.ricker, np.array([width])).reshape(-1)
        X_df[name] = values
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def compute_fft(self, X_df: pd.DataFrame, feature, type):
        name = "_".join([feature, "fft", type])
        fft_transform = np.fft.fft(X_df[feature].values)
        if type == "real":
            values = np.real(fft_transform)
        elif type == "imaginary":
            values = np.imag(fft_transform)
        else:
            values = np.abs(fft_transform)
        X_df[name] = values
        X_df = self.fill_missing_values(X_df, name, feature)
        return X_df

    def compute_ratio_pression_magnetique_plasma(self, X_df: pd.DataFrame):
        X_df['PMP_Ratio'] = X_df['Beta'] / X_df['Pdyn']
        return X_df

    def compute_ratio_pression_vitesse_plasma(self, X_df: pd.DataFrame):
        X_df['PVP_Ratio'] = X_df['V'] / X_df['Pdyn']
        return X_df

    def compute_derivative(self, X_df: pd.DataFrame, feature):
        name = "_".join([feature, "derivative"])
        X_df[name] = X_df[feature].diff() / X_df.index.to_series().diff().dt.total_seconds()
        return X_df

    def compute_polynomial_2(self, X_df: pd.DataFrame, feature, ignore_same=False):
        if "_poly_" in feature:
            return X_df

        for other_feature in X_df.columns:
            name = "_".join([feature, "poly", other_feature])
            other_name = "_".join([other_feature, "poly", feature])

            if "_poly_" in other_feature:
                continue
            if ignore_same and feature == other_feature:
                continue
            if name in X_df.columns or other_name in X_df.columns:
                continue

            X_df[name] = X_df[feature] * X_df[other_feature]

        return X_df

    def drop_columns(self, X_df: pd.DataFrame, dropped_columns):
        X_df = X_df.drop(columns=dropped_columns)
        return X_df

    def drop_all_else_columns(self, X_df: pd.DataFrame, kept_columns):
        for feature in X_df.columns:
            if feature not in kept_columns:
                X_df = self.drop_columns(X_df, [feature])
        return X_df

    def transform_new(self, X):

        print("             [*] Preprocessing data")

        columns = ["Beta", ]
        print(f"               - Reducing columns to {columns}")
        X_df = X["Beta"]

        return X_df

    def transform(self, X):
        import warnings
        warnings.filterwarnings('ignore')
        print("               [*] Preprocessing data")

        X_df = X

        smoothed_columns = ["Beta", "RmsBob", "B", "V"]
        print(f"               - Smooth features: {smoothed_columns}")
        for feature in smoothed_columns:
            X_df = self.smooth(X_df, feature, time_window="1h", center=False)

        print(f"               - Computing special parameters")
        X_df = self.compute_derivative(X_df, 'Beta')
        X_df = self.compute_derivative(X_df, 'Bx')
        X_df = self.compute_ratio_pression_magnetique_plasma(X_df)
        X_df = self.compute_ratio_pression_vitesse_plasma(X_df)

        print("               - Counting peaks and height")
        X_df = self.compute_rolling_count_peaks(X_df, "Pdyn", time_window="2h")
        X_df = self.compute_rolling_peaks_height(X_df, "Pdyn", time_window="2h")
        X_df = self.compute_rolling_peaks_height(X_df, "Pdyn", time_window="20h")
        X_df = self.compute_rolling_peaks_height(X_df, "Pdyn", time_window="50h")
        X_df = self.compute_rolling_peaks_height(X_df, "Pdyn", time_window="100h")


        print("               - Rolling variance")
        tabular = ["Beta", "Np", "Np_nl"]
        for feature in tabular:
            X_df = self.compute_rolling_var(X_df, feature, "1h")
            X_df = self.compute_rolling_var(X_df, feature, "5h")
            X_df = self.compute_rolling_var(X_df, feature, "20h")
            X_df = self.compute_rolling_var(X_df, feature, "50h")
            X_df = self.compute_rolling_var(X_df, feature, "100h")

        print("               - Rolling min")
        tabular = ["Beta", "B", "By", "Bz", "Na_nl", "Vx", "PMP_Ratio"]
        for feature in tabular:
            X_df = self.compute_rolling_min(X_df, feature, "1h")
            X_df = self.compute_rolling_min(X_df, feature, "5h")
            X_df = self.compute_rolling_min(X_df, feature, "15h")
            X_df = self.compute_rolling_min(X_df, feature, "50h")
            X_df = self.compute_rolling_min(X_df, feature, "100h")

        print("               - Rolling max")
        tabular = ["Beta", "B", "Np", "Np_nl", "Range F 0", "Range F 1", "Range F 10", "V", "Vth", "RmsBob"]
        for feature in tabular:
            X_df = self.compute_rolling_max(X_df, feature, "1h")
            X_df = self.compute_rolling_max(X_df, feature, "5h")
            X_df = self.compute_rolling_max(X_df, feature, "23h")
            X_df = self.compute_rolling_max(X_df, feature, "50h")
            X_df = self.compute_rolling_max(X_df, feature, "100h")

        print("               - CWT")
        X_df = self.compute_cwt(X_df, "Beta", width=20)
        X_df = self.compute_cwt(X_df, "Beta", width=10)
        X_df = self.compute_cwt(X_df, "Beta", width=2)
        X_df = self.compute_cwt(X_df, "Vth", width=20)

        print("               - Rolling quantile")
        X_df = self.compute_rolling_quantile(X_df, "Beta", time_window="2h", quantile=0.9)
        X_df = self.compute_rolling_quantile(X_df, "Beta", time_window="2h", quantile=0.7)
        X_df = self.compute_rolling_quantile(X_df, "Beta", time_window="2h", quantile=0.2)
        X_df = self.compute_rolling_quantile(X_df, "Range F 11", time_window="2h", quantile=0.2)
        X_df = self.compute_rolling_quantile(X_df, "Beta", time_window="2h", quantile=0.9)
        X_df = self.compute_rolling_quantile(X_df, "RmsBob", time_window="2h", quantile=0.1)
        X_df = self.compute_rolling_quantile(X_df, "Vth", time_window="2h", quantile=0.7)
        X_df = self.compute_rolling_quantile(X_df, "Vth", time_window="2h", quantile=0.1)

        print("               - Rolling energy")
        X_df = self.compute_rolling_energy(X_df, "Beta", time_window="2h")
        X_df = self.compute_rolling_energy(X_df, "Vth", time_window="2h")

        print("               - Rolling median")
        X_df = self.compute_rolling_median(X_df, "Range F 11", time_window="2h")
        X_df = self.compute_rolling_median(X_df, "Vth", time_window="2h")

        print("               - Time lags")
        for feature in ["Beta", "RmsBob", "Vx", "Range F 9", "Beta_1h_max"]:
            X_df = self.compute_feature_lag(X_df, feature, -1)
            X_df = self.compute_feature_lag(X_df, feature, -5)
            X_df = self.compute_feature_lag(X_df, feature, -10)
            X_df = self.compute_feature_lag(X_df, feature, -20)
            X_df = self.compute_feature_lag(X_df, feature, -50)
            X_df = self.compute_feature_lag(X_df, feature, -100)
            X_df = self.compute_feature_lag(X_df, feature, 1)
            X_df = self.compute_feature_lag(X_df, feature, 5)
            X_df = self.compute_feature_lag(X_df, feature, 10)
            X_df = self.compute_feature_lag(X_df, feature, 20)
            X_df = self.compute_feature_lag(X_df, feature, 50)
            X_df = self.compute_feature_lag(X_df, feature, 100)

        print("               - Filling missing values")
        X_df = X_df.fillna(method='ffill').fillna(method='bfill')

        print("               [*] Done! Running a few last filters and fitting/predicting...")


        return X_df


def get_preprocessing():
    return preprocessing.StandardScaler(), preprocessing.MinMaxScaler()


from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin


class EnsembleClassifier(MultiOutputMixin, ClassifierMixin, BaseEstimator):

    def __init__(self, moving_avg=6, smoothing_threshold=0.5, random_state=None):
        self.models = []
        self.random_state = random_state
        self.moving_avg = moving_avg
        self.smoothing_threshold = smoothing_threshold
        self.add_xgboost(
            weight_minority_class=1,
            max_depth=4,
            validation_fraction=0.1,
        )
        self.add_xgboost(
            weight_minority_class=2.1,
            max_depth=4,
            validation_fraction=0.4,
        )
        self.add_xgboost(
            weight_minority_class=2.8,
            max_depth=4,
            validation_fraction=0.1,
        )

    def add_xgboost(self, weight_minority_class=2.0, max_depth=2, learning_rate=10e-2, validation_fraction=0.5):
        classifier = HistGradientBoostingClassifier(max_iter=200,
                                                    loss='log_loss',
                                                    max_depth=max_depth,
                                                    learning_rate=learning_rate,
                                                    l2_regularization=2,
                                                    early_stopping=True,
                                                    validation_fraction=validation_fraction,
                                                    tol=10e-3,
                                                    class_weight={0: 1, 1: weight_minority_class},
                                                    random_state=self.random_state)
        self.models.append(classifier)

    def fit(self, X, y, sample_weight=None):
        for model in self.models:
            model.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        probas = np.zeros((X.shape[0], len(self.classes_)))

        for model in self.models:
            predictions = model.predict_proba(X)
            for c in range(len(self.classes_)):
                probas[:, c] += predictions[:, c] / len(self.models)

        probas = probas / probas.sum(axis=1)[:, np.newaxis]
        return probas

    def predict(self, X):
        predictions = np.argmax(self.predict_proba(X), axis=1)
        predictions = pd.DataFrame(data=predictions).rolling(self.moving_avg).mean().ffill().bfill().values
        predictions[predictions > self.smoothing_threshold] = 1
        predictions[predictions <= self.smoothing_threshold] = 0
        return predictions


def get_estimator() -> Pipeline:
    feature_extractor = FeatureExtractor()

    classifier = EnsembleClassifier(moving_avg=10, smoothing_threshold=0.7)

    pipe = make_pipeline(
        feature_extractor,
        *get_preprocessing(),
        classifier
    )
    return pipe

