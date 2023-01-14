from utils import fetch_data, run_estimator

import test_environment.classifiers.randomforest.estimator as randomforest
import test_environment.classifiers.testclassifier.estimator as current
import test_environment.classifiers.baggingclassifier.estimator as bagging
import test_environment.classifiers.gradientboosting.estimator as gradientboosting
import test_environment.classifiers.cnn.estimator as cnn

# Fetching data
data_train, label_train, data_test, label_test = fetch_data(1)
data = (data_train, label_train, data_test, label_test)

# Current best: RandomForest

# run_estimator(starting_kit.get_estimator(), "Starting kit")
# run_estimator(bagging.get_estimator(), "Bagging HistBoostingTree", *data)
# run_estimator(randomforest.get_estimator(), "RandomForest", *data)
run_estimator(current.get_estimator(), "Markov Random Forest (12)", *data)
# run_estimator(cnn.get_estimator(), "CNN", *data)
