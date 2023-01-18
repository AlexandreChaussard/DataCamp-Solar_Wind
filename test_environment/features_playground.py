import pandas as pd
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

import problem
from problem import turn_prediction_to_event_list

data = problem.get_train_data(path="../")
X_train: pd.DataFrame = data[0]
y_train: pd.DataFrame = data[1]

events = turn_prediction_to_event_list(y_train)


def extract_located_area(event_index, delta=70):
    start = pd.to_datetime(events[event_index].begin)
    end = pd.to_datetime(events[event_index].end)
    X_df = X_train[(start - pd.Timedelta(hours=delta)):(end + pd.Timedelta(hours=delta))]
    y = y_train[(start - pd.Timedelta(hours=delta)):(end + pd.Timedelta(hours=delta))]
    return X_df, y


def search_features(event_index, column_id, delta=70):
    print(f"[Feature Extraction] Looking for features at event {event_index} on column {column_id} with delta {delta}")
    X_df, y = extract_located_area(event_index, delta)
    print(f"                     * Window size is {X_df.shape[0]}")
    print(f"                     * Extracting features...")
    extracted_features = extract_features(X_df, column_id=column_id)
    print(f"                     * Imputing values...")
    extracted_features = impute(extracted_features)
    print(f"                     * Selecting features...")
    extracted_features = select_features(extracted_features, y.values())
    print(f"                     * Found: {extracted_features}")
    return extracted_features


features = search_features(20, column_id="Beta", delta=80)
print(features)
