# %%
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
from hdbscan.prediction import membership_vector
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from category_encoders import CatBoostEncoder
from copy import deepcopy
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

pd.set_option('display.max_columns', None)

# %%
flight_data_train = pd.read_csv('./flight_data_train_ts.csv')
flight_data_test = pd.read_csv('./flight_data_test_ts.csv')

# %%
flight_data_train['scheduledoffblocktime'] = pd.to_datetime(flight_data_train['scheduledoffblocktime'])
flight_data_test['scheduledoffblocktime'] = pd.to_datetime(flight_data_test['scheduledoffblocktime'])

# flight_data_train.sort_values(by='scheduledoffblocktime', inplace=True)
# flight_data_test.sort_values(by='scheduledoffblocktime', inplace=True)

# %% [markdown]
# Feature Selection

# %%
def chi_2(df, x, y):
    # Create a contingency table
    contingency_table = pd.crosstab(df[x], df[y])

    # Perform the Chi-Square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-Square Statistic for {x} and {y}: {chi2}, p-value: {p}, dof:{dof}")

# %%
def cramers_v(df, x, y):
    # Create a contingency table
    contingency_table = pd.crosstab(df[x], df[y])
    
    # Perform the Chi-Square test
    chi2, p, _, _ = chi2_contingency(contingency_table)
    
    # Calculate Cramer's V
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cor = np.sqrt(chi2 / (n * min_dim))
    print(f"Cramer's V  for {x} and {y}: {cor}, p-value: {p}")

# %%
flight_data_train.columns

# %%
chi_2(flight_data_train, 'publicgatenumber', 'finalflightstatus')
cramers_v(flight_data_train, 'publicgatenumber', 'finalflightstatus')

# %%
chi_2(flight_data_train, 'destination_iata', 'finalflightstatus')
cramers_v(flight_data_train, 'destination_iata', 'finalflightstatus')

# %%
chi_2(flight_data_train, 'aircraft_iata', 'finalflightstatus')
cramers_v(flight_data_train, 'aircraft_iata', 'finalflightstatus')

# %%
chi_2(flight_data_train, 'airlinecode_iata', 'finalflightstatus')
cramers_v(flight_data_train, 'airlinecode_iata', 'finalflightstatus')

# %%
# columns_to_drop = ['publicgatenumber']

# flight_data_train.drop(columns=columns_to_drop, axis=1, inplace=True)
# flight_data_test.drop(columns=columns_to_drop, axis=1, inplace=True)

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
y_train = y_train.map({'On-Time': 0, 'Delayed':1})
y_test = y_test.map({'On-Time': 0, 'Delayed':1})

# %%
X_train

# %%
# from imblearn.over_sampling import SMOTENC

# smote = SMOTENC(random_state=42, categorical_features=[0, 1, 2, 3, 5, 16, 17])
# print('Original dataset shape %s' % Counter(y_train_cls))
# X_train, y_train_cls = smote.fit_resample(X_train, y_train_cls)
# print('Resampled dataset shape %s' % Counter(y_train_cls))


# %%
import pandas as pd
from category_encoders import CatBoostEncoder

high_cardinality_cols = ['airlinecode_iata', 'destination_iata', 'aircraft_iata', 'publicgatenumber']

catboost_encoder = CatBoostEncoder(cols=high_cardinality_cols, return_df=True)

X_train_encoded = catboost_encoder.fit_transform(X_train, y_train)

X_test_encoded = catboost_encoder.transform(X_test)

X_train = X_train_encoded
X_test = X_test_encoded

# %%
# one_hot_column =  ['skyc1', 'skyc2', 'traffictypecode', 'aircraftterminal', 'airlinecode_iata', 'destination_iata']
one_hot_column =  ['skyc1', 'skyc2', 'traffictypecode', 'aircraftterminal', ]

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
# Create cyclic features for month and day
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

# Drop original features
X_train = X_train.drop(['depart_month', 'depart_day', 'depart_minute', 'depart_dayofweek'], axis=1)
X_test = X_test.drop(['depart_month', 'depart_day', 'depart_minute', 'depart_dayofweek'], axis=1)


# %%
X_train

# %%
# neighbors = NearestNeighbors(n_neighbors=39*2)
# neighbors_fit = neighbors.fit(X_train)
# distances, indices = neighbors_fit.kneighbors(X_train)
# avg_distance = distances.mean(axis=1)
# plt.plot(np.sort(avg_distance))
# plt.show()

# %%
# sc = DBSCAN(eps=1.4, min_samples=37*2, algorithm='kd_tree')
# clusters = pd.DataFrame(sc.fit_predict(X_train), columns=['Cluster'])
# clusters['Cluster'].value_counts()

# %%
param_space = {
    'n_estimators': Integer(50, 200),
    'max_depth': Integer(5, 50),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 5),
    'max_features': Categorical(['sqrt', 'log2']),
    'criterion': Categorical(['gini', 'entropy', 'log_loss'])
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
    n_iter=50,
    cv=5,
    n_jobs=-1,
    n_points=2,
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
    min_cluster_size=3000,      # Increase to avoid microclusters
    min_samples=5,              # Lower to reduce noise points
    cluster_selection_epsilon=0.5,  # Increase to reduce noise points
    cluster_selection_method='eom',  # 'eom' tends to produce more balanced clusters
    prediction_data=True,
    core_dist_n_jobs=-1
)
clusters_train = pd.DataFrame(hdbscan_model.fit_predict(X_train), columns=['Cluster'])
clusters_train['Cluster'].value_counts()

# %%
clusters_test, _ = approximate_predict(hdbscan_model, X_test)
pd.Series(clusters_test).value_counts()

# %%
probs_test = membership_vector(hdbscan_model, X_test.to_numpy())
pd.DataFrame(probs_test).head()


# %%
X_train['cluster'] = clusters_train['Cluster']
X_test['cluster'] = clusters_test


# %%
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
        'n_estimators': Integer(50, 200),
        'max_depth': Integer(5, 50),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 5),
        'max_features': Categorical(['sqrt', 'log2']),
        'criterion': Categorical(['gini', 'entropy', 'log_loss']),
    }

    # Use Bayesian optimization for hyperparameter tuning
    bayes_cv = BayesSearchCV(
        estimator=RandomForestClassifier(),
        search_spaces=param_space,
        n_iter=50,
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


# %%
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
    n_iter=10,
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

# %%
cluster_centers = {c: X_train[X_train['cluster'] == c].drop(columns=['cluster']).mean() for c in rf_models.keys()}

# %%
total_entropy_weight = sum(entropy_weights.values())
for model_cluster in entropy_weights:
    entropy_weights[model_cluster] /= total_entropy_weight

# %%
model_weights_f1

# %%
# Make final predictions on the test set using weighted average
final_predictions_f1_weighted = []
final_predictions_cluster_weighted = []
final_predictions_probability_weighted = []
final_predictions_non_weighted = []
final_predictions_pre_classifier = []
final_predictions_entropy_weighted = []
all_y_true = []
for idx in X_test.index:
    cluster = X_test.loc[idx, 'cluster']
    if cluster == -1:
        # Find nearest cluster using euclidean distance to cluster centroids
        X_point = X_test.drop(columns=['cluster']).loc[idx]
        min_dist = float('inf')
        nearest_cluster = None
        
        for c in rf_models.keys():
            # Calculate distance to precalculated centroid
            centroid = cluster_centers[c]
            dist = np.sqrt(((X_point - centroid) ** 2).sum())
            if dist < min_dist:
                min_dist = dist
                nearest_cluster = c
                
        cluster = nearest_cluster
        
    probabilities_hdbscan = probs_test[idx]
    probabilities_pre_classifier = pre_classifier_probabilities[idx]
    
    votes_weighted_f1 = {}
    votes_weighted_cluster = {}
    votes_probability = {}
    votes_pre_classifier = {}
    votes_entropy_weighted = {}
    for model_cluster, model in rf_models.items():
        prediction = model.predict(X_test.drop(columns=['cluster']).loc[[idx]])[0]
        weight_f1 = model_weights_f1.get(model_cluster, 0)
        if prediction in votes_weighted_f1:
            votes_weighted_f1[prediction] += weight_f1
        else:
            votes_weighted_f1[prediction] = weight_f1

        weight_cluster = model_weights_cluster.get(model_cluster, 0)
        if prediction in votes_weighted_cluster:
            votes_weighted_cluster[prediction] += weight_cluster
        else:
            votes_weighted_cluster[prediction] = weight_cluster
            
        if prediction in votes_probability:
            votes_probability[prediction] += probabilities_hdbscan[model_cluster]
        else:
            votes_probability[prediction] = probabilities_hdbscan[model_cluster]
        
        if prediction in votes_pre_classifier:
            votes_pre_classifier[prediction] += probabilities_pre_classifier[model_cluster]
        else:
            votes_pre_classifier[prediction] = probabilities_pre_classifier[model_cluster]
            
        if prediction in votes_entropy_weighted:
            votes_entropy_weighted[prediction] += entropy_weights[model_cluster]
        else:
            votes_entropy_weighted[prediction] = entropy_weights[model_cluster]
        
        
    model = rf_models[cluster]
    prediction = model.predict(X_test.drop(columns=['cluster']).loc[[idx]])[0]
    final_predictions_non_weighted.append(round(prediction))   
    

    # Final prediction is the weighted average
    final_prediction_f1_weighted = max(votes_weighted_f1, key=votes_weighted_f1.get)
    final_predictions_f1_weighted.append(final_prediction_f1_weighted)
    
    final_prediction_cluster_weighted = max(votes_weighted_cluster, key=votes_weighted_cluster.get)
    final_predictions_cluster_weighted.append(final_prediction_cluster_weighted)
    
    final_prediction_probability_weighted = max(votes_probability, key=votes_probability.get)
    final_predictions_probability_weighted.append(final_prediction_probability_weighted)
    
    final_prediction_pre_classifier = max(votes_pre_classifier, key=votes_pre_classifier.get)
    final_predictions_pre_classifier.append(final_prediction_pre_classifier)
    
    final_prediction_entropy_weighted = max(votes_entropy_weighted, key=votes_entropy_weighted.get)
    final_predictions_entropy_weighted.append(final_prediction_entropy_weighted)
    
    all_y_true.append(y_test.loc[idx])


# %%
# Calculate overall metrics
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

# %%
# final_predictions_entropy_weighted = []
# all_y_true = []
# for idx in X_test.index:
    
#     votes_entropy_weighted = {}
#     for model_cluster, model in rf_models.items():
#         prediction = model.predict(X_test.drop(columns=['cluster']).loc[[idx]])[0]
        
#         if prediction in votes_entropy_weighted:
#             votes_entropy_weighted[prediction] += entropy_weights[model_cluster]
#         else:
#             votes_entropy_weighted[prediction] = entropy_weights[model_cluster]    

#     # Final prediction is the weighted average
#     final_prediction_entropy_weighted = max(votes_entropy_weighted, key=votes_entropy_weighted.get)
#     final_predictions_entropy_weighted.append(final_prediction_entropy_weighted)
    
#     all_y_true.append(y_test.loc[idx])
    
# overall_accuracy = accuracy_score(all_y_true, final_predictions_entropy_weighted)
# overall_precision = precision_score(all_y_true, final_predictions_entropy_weighted)
# overall_recall = recall_score(all_y_true, final_predictions_entropy_weighted)
# overall_f1 = f1_score(all_y_true, final_predictions_entropy_weighted)

# print("Overall Metrics (Entropy Weighted):")
# print(f"Overall Accuracy: {overall_accuracy}")
# print(f"Overall Precision: {overall_precision}")
# print(f"Overall Recall: {overall_recall}")
# print(f"Overall F1 Score: {overall_f1}")

# %%
# # Make final predictions on the test set using weighted average
# final_predictions_pre_classifier = []
# all_y_true = []
# for idx in X_test.index:
        
    
    
#     votes_pre_classifier = {}
#     for model_cluster, model in rf_models.items():
#         prediction = model.predict(X_test.drop(columns=['cluster']).loc[[idx]])[0]
        
            

#     # Final prediction is the weighted average
#     final_prediction_pre_classifier = max(votes_pre_classifier, key=votes_pre_classifier.get)
#     final_predictions_pre_classifier.append(final_prediction_pre_classifier)
    
#     all_y_true.append(y_test.loc[idx])

# %%
# overall_accuracy = accuracy_score(all_y_true, final_predictions_pre_classifier)
# overall_precision = precision_score(all_y_true, final_predictions_pre_classifier)
# overall_recall = recall_score(all_y_true, final_predictions_pre_classifier)
# overall_f1 = f1_score(all_y_true, final_predictions_pre_classifier)

# print("\nOverall Metrics (Pre-Classifier):")
# print(f"Overall Accuracy: {overall_accuracy}")
# print(f"Overall Precision: {overall_precision}")
# print(f"Overall Recall: {overall_recall}")
# print(f"Overall F1 Score: {overall_f1}")

# %%
# entropy_weights = {}
# for model_cluster, model in rf_models.items():
#     # Calculate the relative error of the sub-network
#     X_cluster = X_train[X_train['cluster'] == model_cluster].drop(columns=['cluster'])
#     y_cluster = y_train.loc[X_cluster.index]
#     predictions = model.predict(X_cluster)
#     relative_errors = (predictions != y_cluster).astype(int)
#     p_error = relative_errors.sum() / len(relative_errors)
        
#     # Calculate the entropy based on relative error
#     entropy = -p_error * np.log(p_error + 1e-9) - (1 - p_error) * np.log(1 - p_error + 1e-9)
#     weight = 1 - entropy
#     entropy_weights[model_cluster] = weight
    
# # Normalize weights


# %%
# 

# %%
# total_weight = sum(entropy_weights.values())
# for model_cluster in entropy_weights:
#     entropy_weights[model_cluster] /= total_weight
# entropy_weights

# %%



