import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN, approximate_predict
from hdbscan.prediction import membership_vector
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from category_encoders import CatBoostEncoder

pd.set_option('display.max_columns', None)


flight_data_train = pd.read_csv('./flight_data_train_ts_wx.csv')
flight_data_test = pd.read_csv('./flight_data_test_ts_wx.csv')

print(f'Data shape: Train: {flight_data_train.shape}, Test: {flight_data_test.shape}')

print('Preprocessing')

flight_data_train['scheduledoffblocktime'] = pd.to_datetime(flight_data_train['scheduledoffblocktime'])
flight_data_test['scheduledoffblocktime'] = pd.to_datetime(flight_data_test['scheduledoffblocktime'])

flight_data_train.sort_values(by='scheduledoffblocktime', inplace=True)
flight_data_test.sort_values(by='scheduledoffblocktime', inplace=True)


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


X_train = flight_data_train.drop(columns=['delay_in_secs', 'finalflightstatus'], axis=1)
X_test = flight_data_test.drop(columns=['delay_in_secs', 'finalflightstatus'], axis=1)

y_train = flight_data_train['finalflightstatus']
y_test = flight_data_test['finalflightstatus']


y_train = y_train.map({'On-Time': 0, 'Delayed':1})
y_test = y_test.map({'On-Time': 0, 'Delayed':1})

# High cardinality columns - CatBoostEncoder
high_cardinality_cols = ['airlinecode_iata', 'destination_iata', 'aircraft_iata', 'publicgatenumber']
catboost_encoder = CatBoostEncoder(cols=high_cardinality_cols, return_df=True)
X_train = catboost_encoder.fit_transform(X_train, y_train)
X_test = catboost_encoder.transform(X_test)

# One-hot encoding
one_hot_column =  ['skyc1', 'skyc2', 'traffictypecode', 'aircraftterminal', 'wxcodes']
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoded = ohe.fit_transform(X_train[one_hot_column])
ohe_new_columns = ohe.get_feature_names_out(one_hot_column)
encoded_df = pd.DataFrame(encoded, columns=ohe_new_columns)
X_train = pd.concat([X_train.drop(columns=one_hot_column), encoded_df], axis=1)
encoded = ohe.transform(X_test[one_hot_column])
encoded_df = pd.DataFrame(encoded, columns=ohe_new_columns)
X_test = pd.concat([X_test.drop(columns=one_hot_column), encoded_df], axis=1)

# Numerical columns - StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

print('Training baseline model')

param_space = {
    'n_estimators': Integer(100, 300),
    'max_depth': Integer(10, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Categorical(['sqrt', 'log2']),
    'criterion': Categorical(['gini', 'entropy'])
}

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Print class distribution before and after SMOTE
print("Original class distribution:")
print(pd.Series(y_train).value_counts())
print("\nBalanced class distribution:")
print(pd.Series(y_train_balanced).value_counts())

bayes_cv = BayesSearchCV(
    estimator=RandomForestClassifier(),
    search_spaces=param_space,
    n_iter=5,
    cv=5,
    n_jobs=-1,
    n_points=2,
    scoring='f1',
    random_state=42
)
bayes_cv.fit(X_train_balanced, y_train_balanced)
best_rf_no_cluster = bayes_cv.best_estimator_

y_pred = best_rf_no_cluster.predict(X_test)
accuracy_no_cluster = accuracy_score(y_test, y_pred)
precision_no_cluster = precision_score(y_test, y_pred)
recall_no_cluster = recall_score(y_test, y_pred)
f1_no_cluster = f1_score(y_test, y_pred)

print(f"Performance without clustering:")
print(f"Accuracy: {accuracy_no_cluster}")
print(f"Precision: {precision_no_cluster}")
print(f"Recall: {recall_no_cluster}")
print(f"F1 Score: {f1_no_cluster}")
print("==========================================")
print()

print('Training HDBSCAN')
hdbscan_model = HDBSCAN(
    min_cluster_size=5000,      # Increase to avoid microclusters
    min_samples=10,              # Lower to reduce noise points
    cluster_selection_epsilon=0.5,  # Increase to reduce noise points
    cluster_selection_method='eom',  # 'eom' tends to produce more balanced clusters
    prediction_data=True,
    core_dist_n_jobs=-1
)
clusters_train = pd.DataFrame(hdbscan_model.fit_predict(X_train), columns=['Cluster'])
clusters_train['Cluster'].value_counts()

clusters_test, _ = approximate_predict(hdbscan_model, X_test)
pd.Series(clusters_test).value_counts()

probs_test = membership_vector(hdbscan_model, X_test.to_numpy())
pd.DataFrame(probs_test).head()

X_train['cluster'] = clusters_train['Cluster']
X_test['cluster'] = clusters_test

print('Training RFs for each cluster')
rf_models = {}
metrics = {}
model_weights_f1 = {}
model_weights_cluster = {}
entropy_weights = {}
for cluster in np.unique(clusters_train):
    if cluster == -1:  # Ignore noise points (label -1)
        continue
    
    print(f'Cluster {cluster}')

    # Subset the training and test data for the cluster
    # Get cluster data
    X_cluster = X_train[X_train['cluster'] == cluster].drop(columns=['cluster'])
    y_cluster = y_train.loc[X_cluster.index]
    
    X_test_cluster = X_test[X_test['cluster'] == cluster].drop(columns=['cluster'])
    y_test_cluster = y_test.loc[X_test_cluster.index]
    
    # Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_cluster, y_cluster)

# Print class distribution before and after SMOTE
    print("Original class distribution:")
    print(pd.Series(y_cluster).value_counts())
    print("\nBalanced class distribution:")
    print(pd.Series(y_train_balanced).value_counts())

    # Define objective function for hyperopt
    param_space = {
        'n_estimators': Integer(100, 300),
        'max_depth': Integer(10, 50),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 10),
        'max_features': Categorical(['sqrt', 'log2']),
        'criterion': Categorical(['gini', 'entropy'])
    }

    # Use Bayesian optimization for hyperparameter tuning
    bayes_cv = BayesSearchCV(
        estimator=RandomForestClassifier(),
        search_spaces=param_space,
        n_iter=5,
        cv=5,
        n_jobs=-1,
        n_points=2,
        scoring='f1',
        random_state=42
    )
    bayes_cv.fit(X_train_balanced, y_train_balanced)
    best_rf = bayes_cv.best_estimator_
    rf_models[cluster] = best_rf
    
    # Cluster weighted
    cluster_weight = len(y_cluster) / len(y_train)
    model_weights_cluster[cluster] = cluster_weight
    
    # f1 weighted
    model_weights_f1[cluster] = bayes_cv.best_score_
    
    # entropy weighted
    predictions = best_rf.predict(X_cluster)
    relative_errors = (predictions != y_cluster).astype(int)
    p_error = relative_errors.sum() / len(relative_errors)
    entropy = -p_error * np.log(p_error + 1e-9) - (1 - p_error) * np.log(1 - p_error + 1e-9)
    weight = 1 - entropy
    entropy_weights[cluster] = weight
    
    print(f"Cluster {cluster} best f1 score: {bayes_cv.best_score_}")



print('Training Pre-Classifier')
param_space = {
    'n_estimators': Integer(100, 300),
    'max_depth': Integer(10, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Categorical(['sqrt', 'log2']),
    'criterion': Categorical(['gini', 'entropy'])
}
bayes_cv = BayesSearchCV(
    estimator=RandomForestClassifier(),
    search_spaces=param_space,
    n_iter=5,
    cv=3,
    n_jobs=-1,
    n_points=2,
    scoring='f1_weighted',
    random_state=42
)
bayes_cv.fit(X_train[X_train['cluster'] != -1].drop(columns=['cluster']), X_train['cluster'][X_train['cluster'] != -1])
print(bayes_cv.best_score_)
pre_classifier = bayes_cv.best_estimator_
pre_classifier_probabilities = pre_classifier.predict_proba(X_test.drop(columns=['cluster']))

cluster_centers = {c: X_train[X_train['cluster'] == c].drop(columns=['cluster']).mean() for c in rf_models.keys()}

total_entropy_weight = sum(entropy_weights.values())
for model_cluster in entropy_weights:
    entropy_weights[model_cluster] /= total_entropy_weight

# Make final predictions on the test set using weighted average
print('Making final predictions')
final_predictions_f1_weighted = []
final_predictions_cluster_weighted = []
final_predictions_probability_weighted = []
final_predictions_non_weighted = []
final_predictions_pre_classifier = []
final_predictions_entropy_weighted = []
all_y_true = []

# Precompute X_test without 'cluster' column for efficiency
X_test_no_cluster = X_test.drop(columns=['cluster'])

# Precompute distances to cluster centroids for all points
distances_to_centroids = {
    idx: {c: np.linalg.norm(X_test_no_cluster.loc[idx] - cluster_centers[c]) for c in rf_models.keys()}
    for idx in X_test.index
}

for idx in X_test.index:
    cluster = X_test.loc[idx, 'cluster']
    if cluster == -1:
        # Find nearest cluster using precomputed distances
        nearest_cluster = min(distances_to_centroids[idx], key=distances_to_centroids[idx].get)
        cluster = nearest_cluster

    probabilities_hdbscan = probs_test[idx]
    probabilities_pre_classifier = pre_classifier_probabilities[idx]

    votes_weighted_f1 = {}
    votes_weighted_cluster = {}
    votes_probability = {}
    votes_pre_classifier = {}
    votes_entropy_weighted = {}

    X_test_point = X_test_no_cluster.loc[[idx]]
    for model_cluster, model in rf_models.items():
        prediction = model.predict(X_test_point)[0]
        weight_f1 = model_weights_f1.get(model_cluster, 0)
        votes_weighted_f1[prediction] = votes_weighted_f1.get(prediction, 0) + weight_f1

        weight_cluster = model_weights_cluster.get(model_cluster, 0)
        votes_weighted_cluster[prediction] = votes_weighted_cluster.get(prediction, 0) + weight_cluster

        votes_probability[prediction] = votes_probability.get(prediction, 0) + probabilities_hdbscan[model_cluster]
        votes_pre_classifier[prediction] = votes_pre_classifier.get(prediction, 0) + probabilities_pre_classifier[model_cluster]
        votes_entropy_weighted[prediction] = votes_entropy_weighted.get(prediction, 0) + entropy_weights[model_cluster]

    model = rf_models[cluster]
    prediction = model.predict(X_test_point)[0]
    final_predictions_non_weighted.append(prediction)

    # Final prediction is the weighted average
    final_predictions_f1_weighted.append(max(votes_weighted_f1, key=votes_weighted_f1.get))
    final_predictions_cluster_weighted.append(max(votes_weighted_cluster, key=votes_weighted_cluster.get))
    final_predictions_probability_weighted.append(max(votes_probability, key=votes_probability.get))
    final_predictions_pre_classifier.append(max(votes_pre_classifier, key=votes_pre_classifier.get))
    final_predictions_entropy_weighted.append(max(votes_entropy_weighted, key=votes_entropy_weighted.get))

    all_y_true.append(y_test.loc[idx])



# Calculate overall metrics
print(f"Performance without clustering:")
print(f"Accuracy: {accuracy_no_cluster}")
print(f"Precision: {precision_no_cluster}")
print(f"Recall: {recall_no_cluster}")
print(f"F1 Score: {f1_no_cluster}")
print("==========================================")
print()
##########################
# Weighted Average F1 Ensemble
##########################
overall_accuracy = accuracy_score(all_y_true, final_predictions_f1_weighted)
overall_precision = precision_score(all_y_true, final_predictions_f1_weighted)
overall_recall = recall_score(all_y_true, final_predictions_f1_weighted)
overall_f1 = f1_score(all_y_true, final_predictions_f1_weighted)

print("\nOverall Metrics (Weighted Average F1 Ensemble):")
print(f"Overall Accuracy: {overall_accuracy}")
print(f"Overall Precision: {overall_precision}")
print(f"Overall Recall: {overall_recall}")
print(f"Overall F1 Score: {overall_f1}")

##########################
# Weighted Average Cluster Size Ensemble
##########################
overall_accuracy = accuracy_score(all_y_true, final_predictions_cluster_weighted)
overall_precision = precision_score(all_y_true, final_predictions_cluster_weighted)
overall_recall = recall_score(all_y_true, final_predictions_cluster_weighted)
overall_f1 = f1_score(all_y_true, final_predictions_cluster_weighted)

print("\nOverall Metrics (Weighted Average Cluster Size Ensemble):")
print(f"Overall Accuracy: {overall_accuracy}")
print(f"Overall Precision: {overall_precision}")
print(f"Overall Recall: {overall_recall}")
print(f"Overall F1 Score: {overall_f1}")

##########################
# Weighted Average Probability Ensemble
##########################
overall_accuracy = accuracy_score(all_y_true, final_predictions_probability_weighted)
overall_precision = precision_score(all_y_true, final_predictions_probability_weighted)
overall_recall = recall_score(all_y_true, final_predictions_probability_weighted)
overall_f1 = f1_score(all_y_true, final_predictions_probability_weighted)

print("\nOverall Metrics (Weighted Average Probability Ensemble):")
print(f"Overall Accuracy: {overall_accuracy}")
print(f"Overall Precision: {overall_precision}")
print(f"Overall Recall: {overall_recall}")
print(f"Overall F1 Score: {overall_f1}")

##########################
# Non-Weighted Ensemble
##########################
overall_accuracy = accuracy_score(all_y_true, final_predictions_non_weighted)
overall_precision = precision_score(all_y_true, final_predictions_non_weighted)
overall_recall = recall_score(all_y_true, final_predictions_non_weighted)
overall_f1 = f1_score(all_y_true, final_predictions_non_weighted)

print("\nOverall Metrics (Non-Weighted):")
print(f"Overall Accuracy: {overall_accuracy}")
print(f"Overall Precision: {overall_precision}")
print(f"Overall Recall: {overall_recall}")
print(f"Overall F1 Score: {overall_f1}")

##########################
# Pre-Classifier
##########################
overall_accuracy = accuracy_score(all_y_true, final_predictions_pre_classifier)
overall_precision = precision_score(all_y_true, final_predictions_pre_classifier)
overall_recall = recall_score(all_y_true, final_predictions_pre_classifier)
overall_f1 = f1_score(all_y_true, final_predictions_pre_classifier)

print("\nOverall Metrics (Pre-Classifier):")
print(f"Overall Accuracy: {overall_accuracy}")
print(f"Overall Precision: {overall_precision}")
print(f"Overall Recall: {overall_recall}")
print(f"Overall F1 Score: {overall_f1}")

##########################
# Entropy Weighted
##########################
overall_accuracy = accuracy_score(all_y_true, final_predictions_entropy_weighted)
overall_precision = precision_score(all_y_true, final_predictions_entropy_weighted)
overall_recall = recall_score(all_y_true, final_predictions_entropy_weighted)
overall_f1 = f1_score(all_y_true, final_predictions_entropy_weighted)

print("\nOverall Metrics (Entropy Weighted):")
print(f"Overall Accuracy: {overall_accuracy}")
print(f"Overall Precision: {overall_precision}")
print(f"Overall Recall: {overall_recall}")
print(f"Overall F1 Score: {overall_f1}")
