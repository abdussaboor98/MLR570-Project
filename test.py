# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
# from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from collections import Counter

# sklearn.set_config(transform_output="pandas")
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# %%
flight_data_train = pd.read_csv('./flight_data_train_ohe.csv')
flight_data_test = pd.read_csv('./flight_data_test_ohe.csv')

# %%
flight_data_train['scheduledoffblocktime'] = pd.to_datetime(flight_data_train['scheduledoffblocktime'])
flight_data_test['scheduledoffblocktime'] = pd.to_datetime(flight_data_test['scheduledoffblocktime'])

# %%
departdatetime = flight_data_train['scheduledoffblocktime'].dt

flight_data_train['depart_day'] = departdatetime.day
flight_data_train['depart_month'] = departdatetime.month
# flight_data_train['depart_year'] = departdatetime.year
flight_data_train['depart_dayofweek'] = departdatetime.dayofweek
flight_data_train['depart_minute'] = departdatetime.hour * 60 + departdatetime.minute
# Test
departdatetime = flight_data_test['scheduledoffblocktime'].dt
flight_data_test['depart_day'] = departdatetime.day
flight_data_test['depart_month'] = departdatetime.month
# flight_data_test['depart_year'] = departdatetime.year
flight_data_test['depart_dayofweek'] = departdatetime.dayofweek
flight_data_test['depart_minute'] = departdatetime.hour * 60 + departdatetime.minute

flight_data_train.drop(columns=['scheduledoffblocktime'], axis=1, inplace=True)
flight_data_test.drop(columns=['scheduledoffblocktime'], axis=1, inplace=True)

# %%
X_train = flight_data_train.drop(columns=['finalflightstatus', 'delay_in_secs', 'publicgatenumber'], axis=1)
X_test = flight_data_test.drop(columns=['finalflightstatus', 'delay_in_secs', 'publicgatenumber'], axis=1)

y_train_cls = flight_data_train['finalflightstatus']
y_test_cls = flight_data_test['finalflightstatus']
y_train_reg = flight_data_train['delay_in_secs']
y_test_reg = flight_data_test['delay_in_secs']

# %%
y_train_cls = y_train_cls.map({'On-Time': 0, 'Delayed':1})
y_test_cls = y_test_cls.map({'On-Time': 0, 'Delayed':1})

# %%
X_train

# %%
one_hot_column =  ['skyc1', 'skyc2', 'traffictypecode', 'aircraftterminal', 'airlinecode_iata', 'destination_iata', 'aircraft_iata']#, 'publicgatenumber']

ohe = OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='infrequent_if_exist')

encoded = ohe.fit_transform(X_train[one_hot_column])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(one_hot_column))
X_train = pd.concat([X_train.drop(columns=one_hot_column), encoded_df], axis=1)

encoded = ohe.transform(X_test[one_hot_column])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(one_hot_column))
X_test = pd.concat([X_test.drop(columns=one_hot_column), encoded_df], axis=1)



# %%
X_train

# %%
numerical_cols = [
    "tmpf",
    "dwpf",
    "relh",
    "drct",
    "sknt",
    "p01i",
    "alti",
    "vsby",
    "skyl1",
    "skyl2",
    # 'depart_day',
    # 'depart_month',
    # 'depart_minute'
]

# %%
# scaler = StandardScaler()
# X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
# X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# %%
scaler = MinMaxScaler(feature_range=(0, 1))
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# %%
X_train

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

# Drop original cyclic features
X_train = X_train.drop(['depart_month', 'depart_day', 'depart_minute', 'depart_dayofweek'], axis=1)
X_test = X_test.drop(['depart_month', 'depart_day', 'depart_minute', 'depart_dayofweek'], axis=1)


# %%
from scipy.stats import f_oneway

# Assuming y_train_cls is your target variable
anova_results = {}
for column in X_train.columns:
    # Perform ANOVA
    f_val, p_val = f_oneway(X_train[column], y_train_cls)
    anova_results[column] = f_val

# Sort the results by F-value
sorted_anova = sorted(anova_results.items(), key=lambda x: x[1], reverse=True)

# %%
# Plotting
plt.figure(figsize=(10, 50))
plt.barh([x[0] for x in sorted_anova], [x[1] for x in sorted_anova], color='skyblue')
plt.xlabel('F-value')
plt.title('ANOVA F-value for each feature against the target')
plt.gca().invert_yaxis()
plt.show()

# %%
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
print('Original dataset shape %s' % Counter(y_train_cls))
X_train, y_train_cls = smote.fit_resample(X_train, y_train_cls)
print('Resampled dataset shape %s' % Counter(y_train_cls))


# %%
X_full = pd.concat([X_train, X_test], axis=0)
X_full

# %%
# neighbors = NearestNeighbors(n_neighbors=39*2)
# neighbors_fit = neighbors.fit(X_train)
# distances, indices = neighbors_fit.kneighbors(X_train)
# avg_distance = distances.mean(axis=1)
# plt.plot(np.sort(avg_distance))
# plt.show()

# %%
# sc = DBSCAN(eps=1, min_samples=39*2, algorithm='kd_tree')
# clusters = pd.DataFrame(sc.fit_predict(X_full), columns=['Cluster'])
# clusters['Cluster'].value_counts()

# %%
hdbscan_model = HDBSCAN(
    min_cluster_size=1000,      # Increase to avoid microclusters
    min_samples=5,              # Lower to reduce noise points
    cluster_selection_epsilon=0.6,  # Increase to reduce noise points
    cluster_selection_method='eom',  # 'eom' tends to produce more balanced clusters
    prediction_data=True
)
clusters_train = pd.DataFrame(hdbscan_model.fit_predict(X_train), columns=['Cluster'])
clusters_train['Cluster'].value_counts()

# %%
from hdbscan import approximate_predict
clusters_test, _ = approximate_predict(hdbscan_model, X_test)

# %%
X_train['cluster'] = clusters_train['Cluster']
X_test['cluster'] = clusters_test

# %%
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

rf_models = {}
metrics = {}
all_y_true = []
all_y_pred = []
model_weights = {}

for cluster in np.unique(clusters_train):
    if cluster == -1:  # Ignore noise points (label -1)
        continue

    # Subset the training and test data for the cluster
    # Get cluster data
    X_cluster = X_train[X_train['cluster'] == cluster].drop(columns=['cluster'])
    y_cls_cluster = y_train_cls.loc[X_cluster.index]
    
    # Split into train and validation sets
    X_train_cluster, X_val_cluster, y_train_cls_cluster, y_val_cls_cluster = train_test_split(
        X_cluster, y_cls_cluster, test_size=0.2, random_state=42
    )
    X_test_cluster = X_test[X_test['cluster'] == cluster].drop(columns=['cluster'])
    y_test_cls_cluster = y_test_cls.loc[X_test_cluster.index]
    
    # Clean data - remove rows with NaN or infinite values
    mask_train = ~(np.isnan(X_train_cluster).any(axis=1) | np.isinf(X_train_cluster).any(axis=1))
    mask_test = ~(np.isnan(X_test_cluster).any(axis=1) | np.isinf(X_test_cluster).any(axis=1))
    
    X_train_cluster = X_train_cluster[mask_train]
    y_train_cls_cluster = y_train_cls_cluster[mask_train]
    X_test_cluster = X_test_cluster[mask_test]
    y_test_cls_cluster = y_test_cls_cluster[mask_test]

    # Define objective function for hyperopt
    param_space = {
        'n_estimators': Integer(50, 200),
        'max_depth': Integer(5, 50),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 5),
        'max_features': Categorical(['sqrt', 'log2']),
        'criterion': Categorical(['gini', 'entropy', 'log_loss'])
    }

    # Use Bayesian optimization for hyperparameter tuning
    bayes_cv = BayesSearchCV(
        estimator=RandomForestClassifier(),
        search_spaces=param_space,
        n_iter=50,
        cv=3,
        n_jobs=-1,
        scoring='f1'
    )
    bayes_cv.fit(X_train_cluster, y_train_cls_cluster)
    best_rf = bayes_cv.best_estimator_
    rf_models[cluster] = best_rf

    # Make predictions on the validation set and evaluate metrics
    y_val_pred = best_rf.predict(X_val_cluster)
    accuracy = accuracy_score(y_val_cls_cluster, y_val_pred)
    precision = precision_score(y_val_cls_cluster, y_val_pred)
    recall = recall_score(y_val_cls_cluster, y_val_pred)
    f1 = f1_score(y_val_cls_cluster, y_val_pred)
    model_weights[cluster] = f1  # Assign weight to the model based on validation F1 score

    # Print metrics
    print(f"Cluster {cluster} Validation Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

# %%
# Normalizing weights to sum to 1
total_weight = sum(model_weights.values())
model_weights = {cluster: weight / total_weight for cluster, weight in model_weights.items()}

# %%
# Make final predictions on the test set using weighted average
final_predictions = []
all_y_true = []
for idx in X_test.index:
    cluster = X_test.loc[idx, 'cluster']
    if cluster == -1:  # If it's noise, you can skip or handle it differently
        continue

    # Get predictions from all models, weighted by their respective scores
    weighted_sum = 0
    total_weight = 0
    for model_cluster, model in rf_models.items():
        weight = model_weights.get(model_cluster, 0)
        # Give highest weight to the model of the cluster that the test data belongs to
        if model_cluster == cluster:
            weight += 1  # Boost the weight of the corresponding cluster model
        prediction = model.predict(X_test.drop(columns=['cluster']).loc[[idx]])[0]
        weighted_sum += prediction * weight
        total_weight += weight

    # Final prediction is the weighted average
    final_prediction = round(weighted_sum / total_weight)
    final_predictions.append(final_prediction)
    all_y_true.append(y_test_cls.loc[idx])


# %%
# Calculate overall metrics
overall_accuracy = accuracy_score(all_y_true, final_predictions)
overall_precision = precision_score(all_y_true, final_predictions)
overall_recall = recall_score(all_y_true, final_predictions)
overall_f1 = f1_score(all_y_true, final_predictions)

print("\nOverall Metrics (Weighted Average Ensemble):")
print(f"Overall Accuracy: {overall_accuracy}")
print(f"Overall Precision: {overall_precision}")
print(f"Overall Recall: {overall_recall}")
print(f"Overall F1 Score: {overall_f1}")

# %%
# from sklearn.manifold import TSNE
# # Perform t-SNE dimensionality reduction
# tsne = TSNE(n_components=2, random_state=42)
# X_tsne = tsne.fit_transform(X_full)


# %%
# # Create scatter plot
# plt.figure(figsize=(12, 8))
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train_cls, cmap='tab20', alpha=0.6)
# plt.colorbar()
# plt.title('t-SNE visualization of HDBSCAN clusters')
# plt.xlabel('t-SNE component 1')
# plt.ylabel('t-SNE component 2')
# plt.show()

# %%



