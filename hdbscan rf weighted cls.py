import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN, approximate_predict
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from category_encoders import CatBoostEncoder
from copy import deepcopy

pd.set_option('display.max_columns', None)

# %%
flight_data_train = pd.read_csv('./flight_data_train_ts.csv')
flight_data_test = pd.read_csv('./flight_data_test_ts.csv')

# %%
flight_data_train['scheduledoffblocktime'] = pd.to_datetime(flight_data_train['scheduledoffblocktime'])
flight_data_test['scheduledoffblocktime'] = pd.to_datetime(flight_data_test['scheduledoffblocktime'])

# %%
departdatetime = flight_data_train['scheduledoffblocktime'].dt

flight_data_train['depart_day'] = departdatetime.day
flight_data_train['depart_month'] = departdatetime.month
flight_data_train['depart_dayofweek'] = departdatetime.dayofweek
flight_data_train['depart_minute'] = departdatetime.hour * 60 + departdatetime.minute

# Test
departdatetime = flight_data_test['scheduledoffblocktime'].dt
flight_data_test['depart_day'] = departdatetime.day
flight_data_test['depart_month'] = departdatetime.month
flight_data_test['depart_dayofweek'] = departdatetime.dayofweek
flight_data_test['depart_minute'] = departdatetime.hour * 60 + departdatetime.minute

flight_data_train.drop(columns=['scheduledoffblocktime'], axis=1, inplace=True)
flight_data_test.drop(columns=['scheduledoffblocktime'], axis=1, inplace=True)

# %%
X_train = flight_data_train.drop(columns=['delay_in_secs', 'finalflightstatus'], axis=1)
X_test = flight_data_test.drop(columns=['delay_in_secs', 'finalflightstatus'], axis=1)

y_train = flight_data_train['finalflightstatus']
y_test = flight_data_test['finalflightstatus']

# %%
y_train = y_train.map({'On-Time': 0, 'Delayed': 1})
y_test = y_test.map({'On-Time': 0, 'Delayed': 1})

# %%
high_cardinality_cols = ['airlinecode_iata', 'destination_iata', 'aircraft_iata', 'publicgatenumber']

catboost_encoder = CatBoostEncoder(cols=high_cardinality_cols, return_df=True)

X_train_encoded = catboost_encoder.fit_transform(X_train, y_train)
X_test_encoded = catboost_encoder.transform(X_test)

X_train = X_train_encoded
X_test = X_test_encoded

# %%
one_hot_column = ['skyc1', 'skyc2', 'traffictypecode', 'aircraftterminal']

ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

encoded = ohe.fit_transform(X_train[one_hot_column])
ohe_new_columns = ohe.get_feature_names_out(one_hot_column)
encoded_df = pd.DataFrame(encoded, columns=ohe_new_columns)
X_train = pd.concat([X_train.drop(columns=one_hot_column), encoded_df], axis=1)

encoded = ohe.transform(X_test[one_hot_column])
encoded_df = pd.DataFrame(encoded, columns=ohe_new_columns)
X_test = pd.concat([X_test.drop(columns=one_hot_column), encoded_df], axis=1)

# %%
numerical_cols = ['tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'alti', 'vsby', 'skyl1', 'skyl2']

# %%
scaler = MinMaxScaler(feature_range=(0, 1))
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# %%
X_train['depart_month_sin'] = np.sin(2 * np.pi * X_train['depart_month'] / 12)
X_train['depart_month_cos'] = np.cos(2 * np.pi * X_train['depart_month'] / 12)
X_train['depart_day_sin'] = np.sin(2 * np.pi * X_train['depart_day'] / 31) 
X_train['depart_day_cos'] = np.cos(2 * np.pi * X_train['depart_day'] / 31)
X_train['depart_dayofweek_sin'] = np.sin(2 * np.pi * X_train['depart_dayofweek'] / 7) 
X_train['depart_dayofweek_cos'] = np.cos(2 * np.pi * X_train['depart_dayofweek'] / 7)
X_train['depart_minute_sin'] = np.sin(2 * np.pi * X_train['depart_minute'] / 1440) 
X_train['depart_minute_cos'] = np.cos(2 * np.pi * X_train['depart_minute'] / 1440)

X_test['depart_month_sin'] = np.sin(2 * np.pi * X_test['depart_month'] / 12)
X_test['depart_month_cos'] = np.cos(2 * np.pi * X_test['depart_month'] / 12)
X_test['depart_day_sin'] = np.sin(2 * np.pi * X_test['depart_day'] / 31)
X_test['depart_day_cos'] = np.cos(2 * np.pi * X_test['depart_day'] / 31)
X_test['depart_dayofweek_sin'] = np.sin(2 * np.pi * X_test['depart_dayofweek'] / 7) 
X_test['depart_dayofweek_cos'] = np.cos(2 * np.pi * X_test['depart_dayofweek'] / 7)
X_test['depart_minute_sin'] = np.sin(2 * np.pi * X_test['depart_minute'] / 1440)
X_test['depart_minute_cos'] = np.cos(2 * np.pi * X_test['depart_minute'] / 1440)

X_train = X_train.drop(['depart_month', 'depart_day', 'depart_minute', 'depart_dayofweek'], axis=1)
X_test = X_test.drop(['depart_month', 'depart_day', 'depart_minute', 'depart_dayofweek'], axis=1)

# %%
param_space = {
    'n_estimators': Integer(50, 200),
    'max_depth': Integer(5, 50),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 5),
    'max_features': Categorical(['sqrt', 'log2']),
    'criterion': Categorical(['gini', 'entropy', 'log_loss'])
}

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("Original class distribution:")
print(pd.Series(y_train).value_counts())
print("\nBalanced class distribution:")
print(pd.Series(y_train_balanced).value_counts())

bayes_cv = BayesSearchCV(
    estimator=RandomForestClassifier(),
    search_spaces=param_space,
    n_iter=50,
    cv=3,
    n_jobs=-1,
    scoring='f1',
    random_state=42
)
bayes_cv.fit(X_train_balanced, y_train_balanced)
best_rf = bayes_cv.best_estimator_

y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Performance without clustering:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("==========================================")
print()

# %%
hdbscan_model = HDBSCAN(
    min_cluster_size=3000,
    min_samples=5,
    cluster_selection_epsilon=0.5,
    cluster_selection_method='eom',
    prediction_data=True,
    core_dist_n_jobs=-1
)
clusters_train = pd.DataFrame(hdbscan_model.fit_predict(X_train), columns=['Cluster'])
clusters_train['Cluster'].value_counts()

# %%
clusters_test, _ = approximate_predict(hdbscan_model, X_test)
pd.Series(clusters_test).value_counts()

# %%
X_train['cluster'] = clusters_train['Cluster']
X_test['cluster'] = clusters_test

# %%
rf_models = {}
metrics = {}
model_weights_f1 = {}
model_weights_cluster = {}

for cluster in np.unique(clusters_train):
    if cluster == -1:
        continue
    
    print(f'Cluster {cluster}')

    X_cluster = X_train[X_train['cluster'] == cluster].drop(columns=['cluster'])
    y_cluster = y_train.loc[X_cluster.index]
    
    X_train_cluster, X_val_cluster, y_train_cluster, y_val_cluster = train_test_split(
        X_cluster, y_cluster, test_size=0.2, random_state=42
    )
    
    X_test_cluster = X_test[X_test['cluster'] == cluster].drop(columns=['cluster'])
    y_test_cluster = y_test.loc[X_test_cluster.index]
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_cluster, y_train_cluster)

    print("Original class distribution:")
    print(pd.Series(y_train_cluster).value_counts())
    print("\nBalanced class distribution:")
    print(pd.Series(y_train_balanced).value_counts())

    param_space = {
        'n_estimators': Integer(50, 200),
        'max_depth': Integer(5, 50),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 5),
        'max_features': Categorical(['sqrt', 'log2']),
        'criterion': Categorical(['gini', 'entropy', 'log_loss']),
    }

    bayes_cv = BayesSearchCV(
        estimator=RandomForestClassifier(),
        search_spaces=param_space,
        n_iter=50,
        cv=3,
        n_jobs=-1,
        scoring='f1',
        random_state=42
    )
    bayes_cv.fit(X_train_balanced, y_train_balanced)
    best_rf = bayes_cv.best_estimator_
    rf_models[cluster] = best_rf
    
    cluster_weight = len(y_cluster) / len(y_train)
    model_weights_cluster[cluster] = cluster_weight

    y_val_pred = best_rf.predict(X_val_cluster)
    accuracy = accuracy_score(y_val_cluster, y_val_pred)
    precision = precision_score(y_val_cluster, y_val_pred)
    recall = recall_score(y_val_cluster, y_val_pred)
    f1 = f1_score(y_val_cluster, y_val_pred)
    model_weights_f1[cluster] = 1 / f1 if f1 != 0 else 0

    print(f"Cluster {cluster} Validation Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("==========================================")
    print()

# %%
model_weights_f1

# %%
model_weights_cluster

# %%
cluster_centers = {c: X_train[X_train['cluster'] == c].drop(columns=['cluster']).mean() for c in rf_models.keys()}
cluster_centers

# %%
final_predictions_f1_weighted = []
final_predictions_cluster_weighted = []
final_predictions_non_weighted = []
all_y_true = []
for idx in X_test.index:
    cluster = X_test.loc[idx, 'cluster']
    if cluster == -1:
        X_point = X_test.drop(columns=['cluster']).loc[idx]
        min_dist = float('inf')
        nearest_cluster = None
        
        for c in rf_models.keys():
            centroid = cluster_centers[c]
            dist = np.sqrt(((X_point - centroid) ** 2).sum())
            if dist < min_dist:
                min_dist = dist
                nearest_cluster = c
                
        cluster = nearest_cluster

    weights_f1_updated = deepcopy(model_weights_f1)
    weights_f1_updated.update({cluster: 1})
    total_weight_f1 = sum(weights_f1_updated.values())
    weights_f1_updated = {cluster: weight / total_weight_f1 for cluster, weight in weights_f1_updated.items()}
    
        
    weighted_sum_f1 = 0
    total_weight_f1 = 0
    weighted_sum_cluster = 0
    total_weight_cluster = 0
    for model_cluster, model in rf_models.items():
        prediction = model.predict(X_test.drop(columns=['cluster']).loc[[idx]])[0]
        
        weight_f1 = weights_f1_updated.get(model_cluster, 0)
        weighted_sum_f1 += prediction * weight_f1
        total_weight_f1 += weight_f1

        weight_cluster = model_weights_cluster.get(model_cluster, 0)
        weighted_sum_cluster += prediction * weight_cluster
        total_weight_cluster += weight_cluster
        
    model = rf_models[cluster]
    prediction = model.predict(X_test.drop(columns=['cluster']).loc[[idx]])[0]
    final_predictions_non_weighted.append(round(prediction))   
    
    final_prediction_f1_weighted = round(weighted_sum_f1 / total_weight_f1)
    final_predictions_f1_weighted.append(final_prediction_f1_weighted)
    final_prediction_cluster_weighted = round(weighted_sum_cluster / total_weight_cluster)
    final_predictions_cluster_weighted.append(final_prediction_cluster_weighted)
    all_y_true.append(y_test.loc[idx])

# %%
overall_accuracy = accuracy_score(all_y_true, final_predictions_f1_weighted)
overall_precision = precision_score(all_y_true, final_predictions_f1_weighted)
overall_recall = recall_score(all_y_true, final_predictions_f1_weighted)
overall_f1 = f1_score(all_y_true, final_predictions_f1_weighted)

print("\nOverall Metrics (Weighted Average F1 Ensemble):")
print(f"Overall Accuracy: {overall_accuracy}")
print(f"Overall Precision: {overall_precision}")
print(f"Overall Recall: {overall_recall}")
print(f"Overall F1 Score: {overall_f1}")

overall_accuracy = accuracy_score(all_y_true, final_predictions_cluster_weighted)
overall_precision = precision_score(all_y_true, final_predictions_cluster_weighted)
overall_recall = recall_score(all_y_true, final_predictions_cluster_weighted)
overall_f1 = f1_score(all_y_true, final_predictions_cluster_weighted)

print("\nOverall Metrics (Weighted Average Cluster Size Ensemble):")
print(f"Overall Accuracy: {overall_accuracy}")
print(f"Overall Precision: {overall_precision}")
print(f"Overall Recall: {overall_recall}")
print(f"Overall F1 Score: {overall_f1}")

overall_accuracy = accuracy_score(all_y_true, final_predictions_non_weighted)
overall_precision = precision_score(all_y_true, final_predictions_non_weighted)
overall_recall = recall_score(all_y_true, final_predictions_non_weighted)
overall_f1 = f1_score(all_y_true, final_predictions_non_weighted)

print("\nOverall Metrics (Non-Weighted):")
print(f"Overall Accuracy: {overall_accuracy}")
print(f"Overall Precision: {overall_precision}")
print(f"Overall Recall: {overall_recall}")
print(f"Overall F1 Score: {overall_f1}")
