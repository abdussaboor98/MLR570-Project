# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors

# sklearn.set_config(transform_output="pandas")
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# %%
flight_data_train = pd.read_csv('./flight_data_train.csv')
flight_data_test = pd.read_csv('./flight_data_test.csv')

# %%
flight_data_train['scheduledoffblocktime'] = pd.to_datetime(flight_data_train['scheduledoffblocktime'])
flight_data_test['scheduledoffblocktime'] = pd.to_datetime(flight_data_test['scheduledoffblocktime'])

# %%
departdatetime = flight_data_train['scheduledoffblocktime'].dt

flight_data_train['depart_day'] = departdatetime.day
flight_data_train['depart_month'] = departdatetime.month
# flight_data_train['depart_year'] = departdatetime.year

flight_data_train['depart_minute'] = departdatetime.hour * 60 + departdatetime.minute
# Test
departdatetime = flight_data_test['scheduledoffblocktime'].dt

flight_data_test['depart_day'] = departdatetime.day
flight_data_test['depart_month'] = departdatetime.month
# flight_data_test['depart_year'] = departdatetime.year

flight_data_test['depart_minute'] = departdatetime.hour * 60 + departdatetime.minute

flight_data_train.drop(columns=['scheduledoffblocktime'], axis=1, inplace=True)
flight_data_test.drop(columns=['scheduledoffblocktime'], axis=1, inplace=True)

# %%
X_train = flight_data_train.drop(columns=['finalflightstatus', 'delay_in_secs'], axis=1)
X_test = flight_data_test.drop(columns=['finalflightstatus', 'delay_in_secs'], axis=1)

y_train_cls = flight_data_train['finalflightstatus']
y_test_cls = flight_data_test['finalflightstatus']
y_train_reg = flight_data_train['delay_in_secs']
y_test_reg = flight_data_test['delay_in_secs']

# %%
y_train_cls = y_train_cls.map({'On-Time': 0, 'Delayed':1})
y_test_cls = y_test_cls.map({'On-Time': 0, 'Delayed':1})

# %%
string_columns = X_train.select_dtypes(include=['object', 'string']).columns
string_columns

# %%
X_train

# %%
# le = LabelEncoder()
# for col in string_columns:
#     X_train[col] = le.fit_transform(X_train[col])
#     X_test[col] = X_test[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# %%
import pandas as pd
from category_encoders import CatBoostEncoder

high_cardinality_cols = ['flightnumber', 'airlinecode_iata', 
                         'destination_iata', 'aircraft_iata', 'aircraftparkingposition', 
                         'publicgatenumber',]

catboost_encoder = CatBoostEncoder(cols=high_cardinality_cols, return_df=True)

X_train_encoded = catboost_encoder.fit_transform(X_train, y_train_cls)

X_test_encoded = catboost_encoder.transform(X_test)

X_train = X_train_encoded
X_test = X_test_encoded

# %%
X_train['traffictypecode'].value_counts()

# %%
one_hot_column =  ['skyc1', 'skyc2', 'traffictypecode', 'aircraftterminal']

ohe = OneHotEncoder(drop='first', sparse_output=False)

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
    'depart_day',
    'depart_month',
    'depart_minute'
]

# %%
# scaler = StandardScaler()
# X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
# X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# %%
scaler = MinMaxScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# %%
X_train

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

# Plotting
plt.figure(figsize=(10, 6))
plt.barh([x[0] for x in sorted_anova], [x[1] for x in sorted_anova], color='skyblue')
plt.xlabel('F-value')
plt.title('ANOVA F-value for each feature against the target')
plt.gca().invert_yaxis()
plt.show()

# %%
# from imblearn.over_sampling import SMOTE
# from collections import Counter
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# def print_metrics(y_test, y_pred):
#     # Accuracy
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {accuracy:.2f}")
    
#     # Precision (weighted for multi-class)
#     precision = precision_score(y_test, y_pred)
#     print(f"Precision: {precision:.2f}")
    
#     # Recall (weighted for multi-class)
#     recall = recall_score(y_test, y_pred)
#     print(f"Recall: {recall:.2f}")
    
#     # F1 Score (weighted for multi-class)
#     f1 = f1_score(y_test, y_pred)
#     print(f"F1 Score: {f1:.2f}")

# %%
# rf_cls = RandomForestClassifier(n_estimators=100, max_depth=20)
# print('Original dataset shape %s' % Counter(y_train_cls))

# sm = SMOTE(random_state=42)
# X_balanced, y_balanced = sm.fit_resample(X_train, y_train_cls)
# print('Resampled dataset shape %s' % Counter(y_balanced))

# rf_cls.fit(X_balanced, y_balanced)

# y_pred_cls = rf_cls.predict(X_test)
# print_metrics(y_test_cls, y_pred_cls)

# %%
# print_metrics(y_test_cls, y_pred_cls)

# %%
X_full = pd.concat([X_train, X_test], axis=0)
X_full

# %%
# neighbors = NearestNeighbors(n_neighbors=36*2)
# neighbors_fit = neighbors.fit(X_train)
# distances, indices = neighbors_fit.kneighbors(X_train)
# avg_distance = distances.mean(axis=1)
# plt.plot(np.sort(avg_distance))
# plt.show()

# %%
sc = DBSCAN(eps=0.6, min_samples=36*2, algorithm='kd_tree')
clusters = pd.DataFrame(sc.fit_predict(X_full), columns=['Cluster'])
print(clusters['Cluster'].value_counts())

# %%
# sc = HDBSCAN()
# clusters = pd.DataFrame(sc.fit_predict(X_full), columns=['Cluster'])


# %%
# clusters['Cluster'].value_counts()


