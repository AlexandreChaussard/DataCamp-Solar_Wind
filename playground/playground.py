import problem
import submissions.randomforest.estimator as randomforest_estimator

from sklearn.metrics import classification_report, confusion_matrix

# Fetching data
print("[Data] Fetching dataset...")
X_train, y_train = problem.get_train_data(path="../")
X_test, y_test = problem.get_test_data(path="../")


# Random forest estimator #

# Instantiating pipeline
print("[Pipeline] * Instantiating pipeline")
estimator = randomforest_estimator.get_estimator()
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
# Printing confusion matrix
print("           * Printing confusion matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)
