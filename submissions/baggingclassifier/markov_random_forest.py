import numpy as np
from sklearn.ensemble import RandomForestClassifier


class MarkovRandomForest(RandomForestClassifier):

    def __init__(
            self,
            memory=1,
            n_estimators=100,
            *,
            criterion='gini',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features='sqrt',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None
    ):
        super().__init__(n_estimators=n_estimators,
                         criterion=criterion,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_features=max_features,
                         max_leaf_nodes=max_leaf_nodes,
                         min_impurity_decrease=min_impurity_decrease,
                         bootstrap=bootstrap,
                         oob_score=oob_score,
                         n_jobs=n_jobs,
                         random_state=random_state,
                         verbose=verbose,
                         warm_start=warm_start,
                         class_weight=class_weight,
                         ccp_alpha=ccp_alpha,
                         max_samples=max_samples)
        self.memory = memory

    def fit(self, X, y, sample_weight=None):
        # Memory matrix
        y_memory = np.zeros((X.shape[0], self.memory))

        # initial memory
        y_current_memory = np.zeros(self.memory)

        # Building memory
        for i in range(X.shape[0]):
            # Saving memory at current index
            y_memory[i, :] = y_current_memory.copy()
            # Refreshing current memory
            y_current_memory[1:] = y_current_memory[0:-1]
            y_current_memory[0] = y[i]

        # Merging memory with global feature vector
        X_df = np.concatenate((X, y_memory), axis=1)

        return super().fit(X_df, y, sample_weight)

    def predict(self, X):
        # Memory matrix
        y_memory = np.zeros((X.shape[0], self.memory))

        # initial memory
        y_current_memory = np.zeros(self.memory)

        for i in range(X.shape[0]):
            # Adding current memory to the global memory
            y_memory[i, :] = y_current_memory.copy()

            # Building current vector
            X_current = X[i]
            # Adding memory of previous predictions
            X_memory = np.concatenate((X_current, y_current_memory)).reshape(1, -1)
            # Prediction step
            pred = super().predict(X_memory)
            # Refreshing current memory
            y_current_memory[1:] = y_current_memory[0:-1]
            y_current_memory[0] = pred

        # Merging memory to the global feature vector to predict
        X_df = np.concatenate((X, y_memory), axis=1)
        return super().predict(X_df)


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X, y = make_classification(n_samples=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

model = MarkovRandomForest(n_estimators=10, max_depth=4)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

