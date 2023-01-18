import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin


class MemoryClassifier(MultiOutputMixin, ClassifierMixin, BaseEstimator):

    def __init__(
            self,
            memory=3,
            classifier=HistGradientBoostingClassifier(max_depth=4),
            characteristic_storm_duration=12,
            residual=0.1,
            random_state=None,
    ):
        self.classifier = classifier
        self.class_distribution = np.array([0.5, 0.5])
        self.memory = memory
        self.residual = residual

        self.classes_ = None

        self.random_state = random_state
        np.random.seed(random_state)

        # characteristic time parameter of a solar storm
        self.characteristic_storm_duration = characteristic_storm_duration

    def solar_storm_decrease(self, duration, residual):
        return residual + (1-residual) * np.exp(-duration / self.characteristic_storm_duration)

    def fit(self, X, y, sample_weight=None):
        # Memory matrix
        y_memory = np.zeros((X.shape[0], self.memory))

        # initial memory
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.class_distribution = np.array([c / counts.sum() for c in counts])
        y_current_memory = np.random.choice(np.unique(y), size=self.memory, p=self.class_distribution) \
            .astype(np.float32)

        # Building memory
        solar_storm_duration = 0  # time criterion of the solar storm
        for i in range(X.shape[0]):
            # Saving memory at current index
            y_memory[i, :] = y_current_memory.copy()
            # Refreshing current memory
            y_current_memory[1:] = y_current_memory[0:-1]

            if y[i] == 1:
                solar_storm_duration += 1
            else:
                solar_storm_duration = 0
            y_current_memory[0] = y[i] * self.solar_storm_decrease(solar_storm_duration, 0)

        # Merging memory with global feature vector
        X_df = np.concatenate((X, y_memory), axis=1)

        return self.classifier.fit(X_df, y, sample_weight)

    def _preprocess_predict(self, X):
        # Memory matrix
        y_memory = np.zeros((X.shape[0], self.memory))

        # initial memory
        y_current_memory = np.random \
            .choice(np.unique(self.classes_), size=self.memory, p=self.class_distribution) \
            .astype(np.float32)

        last_percentage = -1
        solar_storm_duration = 0
        for i in range(X.shape[0]):
            percentage_progress = int(i / X.shape[0] * 100)
            if percentage_progress % 20 == 0 and last_percentage != percentage_progress:
                print(f"            - [{percentage_progress}%] Prediction in progress...")
                last_percentage = percentage_progress
            # Adding current memory to the global memory
            y_memory[i, :] = y_current_memory.copy()

            # Building current vector
            X_current = X[i]
            # Adding memory of previous predictions
            X_memory = np.concatenate((X_current, y_current_memory)).reshape(1, -1)
            # Prediction step
            # pred = self.classifier.predict_proba(X_memory).squeeze()
            pred = self.classifier.predict(X_memory)
            if pred > 0.5:
                solar_storm_duration += 1
            else:
                solar_storm_duration = 0

            # Refreshing current memory
            y_current_memory[1:] = y_current_memory[0:-1]
            # y_current_memory[0] = pred[1]
            y_current_memory[0] = pred * self.solar_storm_decrease(solar_storm_duration, self.residual)

        # Merging memory to the global feature vector to predict
        X_df = np.concatenate((X, y_memory), axis=1)
        print(f"            - [100%] Predictions are done!")
        print(X_df[35])
        return X_df

    def predict_proba(self, X):
        X_df = self._preprocess_predict(X)
        return self.classifier.predict_proba(X_df)

    def predict(self, X):
        X_df = self._preprocess_predict(X)
        return self.classifier.predict(X_df)

    def score(self, X, y, sample_weight=None):
        X_df = self._preprocess_predict(X)
        return self.classifier.score(X_df, y)


def test():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn import preprocessing

    from utils import run_estimator

    X, y = make_classification(n_samples=300)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    def get_preprocessing():
        return preprocessing.QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=1), \
               preprocessing.StandardScaler(), \
               preprocessing.MinMaxScaler()

    classifier = MemoryClassifier(
        memory=5,
        classifier=HistGradientBoostingClassifier(max_depth=5, random_state=2, class_weight='balanced'),
        random_state=2
    )

    pipe = make_pipeline(
        *get_preprocessing(),
        classifier
    )

    run_estimator(pipe, name='Markov Model', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
