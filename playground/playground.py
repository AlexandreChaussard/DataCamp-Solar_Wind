import problem

from sklearn.pipeline import Pipeline

from sklearn.metrics import balanced_accuracy_score

import submissions.randomforest.estimator as randomforest_estimator
import submissions.baggingclassifier.estimator as bagging_estimator
import submissions.starting_kit.estimator as starting_kit

from sklearn.metrics import classification_report, confusion_matrix

# Fetching data
print("[Data] Fetching dataset...")
data_train, label_train = problem.get_train_data(path="../")
data_test, label_test = problem.get_test_data(path="../")


def run_estimator(
        estimator: Pipeline,
        name: str,
        X_train=data_train,
        y_train=label_train,
        X_test=data_test,
        y_test=label_test
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


# run_estimator(starting_kit.get_estimator(), "Starting kit")
run_estimator(bagging_estimator.get_estimator(), "BalancedBagging estimator")
