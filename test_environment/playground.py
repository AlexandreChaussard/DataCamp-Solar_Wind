from utils import fetch_data, run_estimator

import test_environment.classifiers.randomforest.estimator as randomforest
import test_environment.classifiers.markov.estimator as current
import test_environment.classifiers.baggingclassifier.estimator as bagging
import test_environment.classifiers.gradientboosting.estimator as gradientboosting
import test_environment.classifiers.cnn.estimator as cnn
import test_environment.classifiers.ensemble.estimator as ensemble
import test_environment.classifiers.ensemble.memory_estimator as memory_ensemble
import test_environment.classifiers.xgboost.estimator as xgboost
import test_environment.classifiers.mlp.estimator as mlp

# Fetching data
data_train, label_train, data_test, label_test = fetch_data(1)
data = (data_train, label_train, data_test, label_test)

# Current best: RandomForest

# run_estimator(starting_kit.get_estimator(), "Starting kit")
# run_estimator(bagging.get_estimator(), "Bagging HistBoostingTree", *data)
# run_estimator(gradientboosting.get_estimator(), "HistBoostingTree", *data)
# run_estimator(randomforest.get_estimator(), "RandomForest", *data)
# run_estimator(cnn.get_estimator(), "CNN", *data)

# run_estimator(ensemble.get_estimator(), "Custom Ensemble", *data)
run_estimator(memory_ensemble.get_estimator(), "Custom Memory Ensemble", *data)

# run_estimator(xgboost.get_estimator(), "XGBoost", *data)

# run_estimator(mlp.get_estimator(), "Neural network", *data)

# memory = 20
# for memory in [10]:
#    print(f"[***********] MEMORY {memory} [***********]")
#    for time_range in [10, 2, 5]:
#        print(f"     [*@*]-- {memory} memory - {time_range} time range --[*@*]")
#        run_estimator(current.get_estimator(memory=memory, characteristic_storm_duration=3, residual=0.6),
#                      "Markov Random Forest (12)",
#                      *data)
