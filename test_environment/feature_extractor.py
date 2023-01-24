import pandas as pd

import problem

from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.preprocessing as preprocessing
import sklearn.impute as impute

from sklearn.base import BaseEstimator

from scipy import signal
from scipy import stats
import numpy as np


def fetch_data(percentage=1.0):
    print(f"[Data] Fetching {int(percentage * 100)}% of the data...")
    data_train, label_train = problem.get_train_data(path="../")
    data_test, label_test = problem.get_test_data(path="../")

    index_train = int(len(data_train.index) * percentage)
    index_test = int(len(data_test.index) * percentage)
    data_train, label_train = data_train[:index_train], label_train[:index_train]
    data_test, label_test = data_test[:index_test], label_test[:index_test]

    return data_train, label_train, data_test, label_test


def run_estimator(
        estimator: Pipeline,
        name: str,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
):
    print(f"[Pipeline] Running pipepline: {name}")
    # Fitting estimator
    print("           * Fitting the training set")
    estimator.fit(X_train, y_train)
    # Making prediction
    print("           * Predicting on the test set")
    y_pred = estimator.predict(X_test)
    # Print model information
    print("           * Printing report")
    report = classification_report(y_test, y_pred)
    print(report)
    print(f"           * Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred)}")
    # Printing confusion matrix
    print("           * Printing confusion matrix")
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    print(cm)


class FeatureComputer:
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


class FeatureExtractor_ElasticMemory:
    """
    This class is about extracting features with a given memory in time.
    The idea is that I'm gonna feed models with different memory sizes in order to get the most sense out of
    different areas in time.

    The obtained models should be different enough so I can consider a larger "ensemble learning" strategy.
    Then I'm gonna use an aggregation in their probability outputs, and hopefully I'll get the best out of each model.
    """

    def __init__(self, memories, timelags):
        self.memories = memories
        self.timelags = timelags

    def fit(self, X, y):
        return self

    def transform(self, X: pd.DataFrame):
        import warnings
        warnings.filterwarnings('ignore')

        fc = FeatureComputer()

        print(f"               [*] Preprocessing data with memories: {self.memories}")

        X_df = X.copy(deep=True)


        tabular = [
            "Beta",
            "RmsBob",
            "B",
            "V",
            "Vth"
        ]
        print(f"               - Smoothing features")
        for feature in tabular:
            X_df = fc.smooth(X_df, feature, time_window="1h20min", center=False)

        print(f"               - Computing special parameters")
        X_df = fc.compute_derivative(X_df, 'Beta')
        X_df = fc.compute_derivative(X_df, 'Bx')
        X_df = fc.compute_derivative(X_df, 'Range F 10')
        X_df = fc.compute_derivative(X_df, 'V')
        X_df = fc.compute_derivative(X_df, 'Vth')

        print("               - Rolling mean")
        for memory in self.memories:
            X_df = fc.compute_rolling_mean(X_df, 'V_derivative', memory)
            X_df = fc.compute_rolling_mean(X_df, 'Vth_derivative', memory)
            X_df = fc.compute_rolling_mean(X_df, 'Beta', memory)

        print("               - Rolling variance")
        tabular = [
            "Beta",
            "Np",
            "Np_nl",
            "Range F 0",
            "Range F 1",
            "Range F 10",
            "Range F 11",
            "Range F 12",
            "Range F 13",
            "Range F 2",
            "Range F 3",
            "Range F 4",
            "Range F 5",
            "Range F 6",
            "Range F 7",
            "Range F 8",
            "Range F 9"
        ]
        for feature in tabular:
            for memory in self.memories:
                X_df = fc.compute_rolling_var(X_df, feature, memory)

        print("               - Rolling min")
        tabular = [
            "Beta",
            "B",
            "By",
            "Bz",
            "Na_nl",
            "Vx",
            "Range F 10",
            "Range F 10_derivative"
        ]
        for feature in tabular:
            for memory in self.memories:
                X_df = fc.compute_rolling_min(X_df, feature, memory)

        print("               - Rolling max")
        tabular = [
            "Beta",
            "B",
            "Np",
            "Np_nl",
            "Range F 10",
            "Range F 10_derivative",
            "Range F 11",
            "Range F 12",
            "Range F 13",
            "V",
            "Vth",
            "RmsBob"
        ]
        for feature in tabular:
            for memory in self.memories:
                X_df = fc.compute_rolling_max(X_df, feature, memory)

        print("               - CWT")
        X_df = fc.compute_cwt(X_df, "Beta", width=5)
        X_df = fc.compute_cwt(X_df, "Beta", width=2)
        X_df = fc.compute_cwt(X_df, "Vth", width=5)
        X_df = fc.compute_cwt(X_df, "Beta", width=20)
        X_df = fc.compute_cwt(X_df, "Beta", width=10)
        X_df = fc.compute_cwt(X_df, "Vth", width=20)

        print("               - Rolling quantiles")
        X_df = fc.compute_rolling_quantile(X_df, "Beta", time_window="2h", quantile=0.9)
        X_df = fc.compute_rolling_quantile(X_df, "Beta", time_window="2h", quantile=0.7)
        X_df = fc.compute_rolling_quantile(X_df, "Beta", time_window="2h", quantile=0.2)
        X_df = fc.compute_rolling_quantile(X_df, "Range F 11", time_window="2h", quantile=0.2)
        X_df = fc.compute_rolling_quantile(X_df, "Beta", time_window="2h", quantile=0.9)
        X_df = fc.compute_rolling_quantile(X_df, "RmsBob", time_window="2h", quantile=0.1)
        X_df = fc.compute_rolling_quantile(X_df, "Vth", time_window="2h", quantile=0.7)
        X_df = fc.compute_rolling_quantile(X_df, "Vth", time_window="2h", quantile=0.1)

        print("               - Rolling median")
        for memory in self.memories:
            X_df = fc.compute_rolling_median(X_df, "Range F 11", time_window=memory)
            X_df = fc.compute_rolling_median(X_df, "Vth", time_window=memory)

        print("               - Time lags")
        for feature in ["Beta", "RmsBob", "Vx", "Range F 9"]:
            for time_lag in self.timelags:
                X_df = fc.compute_feature_lag(X_df, feature, time_lag)

        print("               - Filling missing values")
        X_df = X_df.fillna(method='ffill').fillna(method='bfill')

        print("               [*] Ready to enter the classifier")

        return X_df


def get_preprocessing():
    return preprocessing.StandardScaler(), preprocessing.MinMaxScaler()


def sliding_label(y: np.array, sliding_window):
    """
    Transforms y (labels in 0,1) in a continuous representation through a sliding window
    that counts the number of ones (CME) seen in the window
    """
    y_reg = np.zeros(len(y))
    for i in range(0, len(y)):
        sliding_index = min(i + sliding_window, len(y) - 1)
        y_reg[i] = y[i:sliding_index].sum()

    y_reg = y_reg
    return y_reg
