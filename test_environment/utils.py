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
        y_test=None
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
        X_df.insert(1, name, number_peaks)
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

    def denoise(self, X_df: pd.DataFrame, feature, threshold):
        fft_transform = np.fft.fft(X_df[feature].values)
        fft_transform[fft_transform < threshold] = 0
        denoised = np.fft.ifft(fft_transform)
        X_df[feature] = denoised
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

        print("               [*] Preprocessing data")

        X_df = X

        smoothed_columns = ["Beta", "RmsBob", "B", "V"]
        print(f"               - Smooth features: {smoothed_columns}")
        for feature in smoothed_columns:
            X_df = self.smooth(X_df, feature, time_window="1h20min", center=False)

        print(f"               - Computing special parameters")
        X_df = self.compute_derivative(X_df, 'Beta')
        X_df = self.compute_ratio_pression_magnetique_plasma(X_df)
        X_df = self.compute_ratio_pression_vitesse_plasma(X_df)

        print("               - Counting peaks and height")
        X_df = self.compute_rolling_count_peaks(X_df, "Pdyn", time_window="2h")
        X_df = self.compute_rolling_peaks_height(X_df, "Pdyn", time_window="2h")

        print("               - Rolling variance")
        X_df = self.compute_rolling_var(X_df, "Beta", "2h")
        X_df = self.compute_rolling_var(X_df, "Beta", "1h")
        X_df = self.compute_rolling_var(X_df, "Beta", "20min")
        X_df = self.compute_rolling_var(X_df, "Np", "40min")
        X_df = self.compute_rolling_var(X_df, "Np_nl", "40min")

        print("               - Rolling min")
        X_df = self.compute_rolling_min(X_df, "By", "30min")
        X_df = self.compute_rolling_min(X_df, "Bz", "30min")
        X_df = self.compute_rolling_min(X_df, "By", "30min")
        X_df = self.compute_rolling_min(X_df, "Na_nl", "30min")
        X_df = self.compute_rolling_min(X_df, "Vx", "40min")
        X_df = self.compute_rolling_min(X_df, "Vy", "40min")
        X_df = self.compute_rolling_min(X_df, "By", "2h")
        X_df = self.compute_rolling_min(X_df, "Bz", "2h")
        X_df = self.compute_rolling_min(X_df, "By", "2h")
        X_df = self.compute_rolling_min(X_df, "Na_nl", "2h")
        X_df = self.compute_rolling_min(X_df, "Vx", "2h")
        X_df = self.compute_rolling_min(X_df, "Vy", "2h")
        X_df = self.compute_rolling_min(X_df, "Pdyn", "2h")

        print("               - Rolling max")
        X_df = self.compute_rolling_max(X_df, "Beta", "1h")
        X_df = self.compute_rolling_max(X_df, "Beta", "30min")
        X_df = self.compute_rolling_max(X_df, "B", "40min")
        X_df = self.compute_rolling_max(X_df, "Np", "40min")
        X_df = self.compute_rolling_max(X_df, "Np_nl", "40min")
        X_df = self.compute_rolling_max(X_df, "Range F 0", "40min")
        X_df = self.compute_rolling_max(X_df, "Range F 1", "40min")
        X_df = self.compute_rolling_max(X_df, "Range F 10", "40min")
        X_df = self.compute_rolling_max(X_df, "V", "40min")
        X_df = self.compute_rolling_max(X_df, "RmsBob", "40min")
        X_df = self.compute_rolling_max(X_df, "Beta", "2h")
        X_df = self.compute_rolling_max(X_df, "B", "2h")
        X_df = self.compute_rolling_max(X_df, "Np", "2h")
        X_df = self.compute_rolling_max(X_df, "Np_nl", "2h")
        X_df = self.compute_rolling_max(X_df, "Range F 0", "2h")
        X_df = self.compute_rolling_max(X_df, "Range F 1", "2h")
        X_df = self.compute_rolling_max(X_df, "Range F 10", "2h")
        X_df = self.compute_rolling_max(X_df, "V", "2h")
        X_df = self.compute_rolling_max(X_df, "Vth", "2h")
        X_df = self.compute_rolling_max(X_df, "RmsBob", "2h")

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
        for feature in ["Beta", "RmsBob", "Vx", "Range F 9", "Beta_30min_max"]:
            X_df = self.compute_feature_lag(X_df, feature, -1)
            X_df = self.compute_feature_lag(X_df, feature, -5)
            X_df = self.compute_feature_lag(X_df, feature, -10)
            X_df = self.compute_feature_lag(X_df, feature, -20)
            X_df = self.compute_feature_lag(X_df, feature, 1)
            X_df = self.compute_feature_lag(X_df, feature, 5)
            X_df = self.compute_feature_lag(X_df, feature, 10)
            X_df = self.compute_feature_lag(X_df, feature, 20)

        # columns = ['Beta_cwt_20', 'Range F 11', 'Range F 7', 'Pdyn_2h_peaks_count', 'Beta_cwt_10', 'Vth', 'Vth_2h_energy', 'RmsBob_20', 'B', 'RmsBob_-20', 'RmsBob_10', 'RmsBob_-10', 'Beta_cwt_2', 'RmsBob_5', 'RmsBob_-5', 'RmsBob_1', 'Beta_20min_var', 'RmsBob', 'RmsBob_-1', 'Na_nl_30min_min', 'By_30min_min', 'Bz_30min_min', 'Beta_1h_var', 'Beta_20', 'Beta_2h_var', 'Beta_-20', 'Beta_10', 'Beta_-10', 'PMP_Ratio', 'Beta_5', 'Beta_-5', 'Range F 11_2h_mean', 'Beta_1', 'Beta', 'Beta_-1', 'Vth_2h_mean', 'Beta_2h_energy', 'Vth_2h_q0.7', 'Pdyn_2h_peaks_height', 'Range F 11_2h_q0.2', 'Beta_30min_max_20', 'Beta_30min_max_-20', 'Beta_30min_max_10', 'Vx_40min_min', 'V_40min_max', 'Np_nl_40min_max', 'Beta_30min_max_-10', 'Beta_30min_max_5', 'B_40min_max', 'Np_40min_max', 'Beta_30min_max_-5', 'Beta_30min_max_1', 'Beta_30min_max', 'Beta_30min_max_-1', 'Vth_2h_q0.1', 'Range F 10_40min_max', 'Vy_40min_min', 'Range F 1_40min_max', 'Beta_2h_q0.7', 'Range F 0_40min_max', 'RmsBob_2h_q0.1', 'Beta_2h_q0.2', 'Beta_2h_q0.9', 'RmsBob_40min_max', 'B_2h_max', 'Np_nl_2h_max', 'Vx_2h_min', 'V_2h_max', 'Na_nl_2h_min', 'Np_2h_max', 'Range F 10_2h_max', 'By_2h_min', 'Bz_2h_min', 'Vth_2h_max', 'Vy_2h_min', 'Beta_1h_max', 'Range F 1_2h_max', 'Beta_2h_max', 'Range F 0_2h_max', 'RmsBob_2h_max']
        # print("               - Limiting columns")
        # X_df = X_df[columns]
        return X_df


def get_preprocessing():
    print("               - Applying preprocessing filters...")
    return impute.SimpleImputer(), \
           preprocessing.PolynomialFeatures(degree=2, interaction_only=True), \
           preprocessing.RobustScaler(), \
           preprocessing.MinMaxScaler(),
