# Import required libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from hdbscan import HDBSCAN, approximate_predict
from hdbscan.prediction import membership_vector
from imblearn.over_sampling import SMOTE
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Define DNN model
class FlightDelayDNN(nn.Module):
    def __init__(self, input_dim):
        super(FlightDelayDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.output(x)
        return self.sigmoid(x)

# Custom Dataset class
class FlightDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# Define training function
def train_model(model, dataloader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")

# Load data
flight_data_train = pd.read_csv('./flight_data_train_ts.csv')
flight_data_test = pd.read_csv('./flight_data_test_ts.csv')

flight_data_train['scheduledoffblocktime'] = pd.to_datetime(flight_data_train['scheduledoffblocktime'])
flight_data_test['scheduledoffblocktime'] = pd.to_datetime(flight_data_test['scheduledoffblocktime'])

# Extract date features
departdatetime = flight_data_train['scheduledoffblocktime'].dt
flight_data_train['depart_day'] = departdatetime.day
flight_data_train['depart_month'] = departdatetime.month
flight_data_train['depart_dayofweek'] = departdatetime.dayofweek
flight_data_train['depart_minute'] = departdatetime.hour * 60 + departdatetime.minute

# Test data
departdatetime = flight_data_test['scheduledoffblocktime'].dt
flight_data_test['depart_day'] = departdatetime.day
flight_data_test['depart_month'] = departdatetime.month
flight_data_test['depart_dayofweek'] = departdatetime.dayofweek
flight_data_test['depart_minute'] = departdatetime.hour * 60 + departdatetime.minute

# Create cyclical features
flight_data_train['depart_month_sin'] = np.sin(2 * np.pi * flight_data_train['depart_month'] / 12)
flight_data_train['depart_month_cos'] = np.cos(2 * np.pi * flight_data_train['depart_month'] / 12)
flight_data_train['depart_day_sin'] = np.sin(2 * np.pi * flight_data_train['depart_day'] / 31)
flight_data_train['depart_day_cos'] = np.cos(2 * np.pi * flight_data_train['depart_day'] / 31)
flight_data_train['depart_dayofweek_sin'] = np.sin(2 * np.pi * flight_data_train['depart_dayofweek'] / 7)
flight_data_train['depart_dayofweek_cos'] = np.cos(2 * np.pi * flight_data_train['depart_dayofweek'] / 7)
flight_data_train['depart_minute_sin'] = np.sin(2 * np.pi * flight_data_train['depart_minute'] / 1440)
flight_data_train['depart_minute_cos'] = np.cos(2 * np.pi * flight_data_train['depart_minute'] / 1440)

flight_data_test['depart_month_sin'] = np.sin(2 * np.pi * flight_data_test['depart_month'] / 12)
flight_data_test['depart_month_cos'] = np.cos(2 * np.pi * flight_data_test['depart_month'] / 12)
flight_data_test['depart_day_sin'] = np.sin(2 * np.pi * flight_data_test['depart_day'] / 31)
flight_data_test['depart_day_cos'] = np.cos(2 * np.pi * flight_data_test['depart_day'] / 31)
flight_data_test['depart_dayofweek_sin'] = np.sin(2 * np.pi * flight_data_test['depart_dayofweek'] / 7)
flight_data_test['depart_dayofweek_cos'] = np.cos(2 * np.pi * flight_data_test['depart_dayofweek'] / 7)
flight_data_test['depart_minute_sin'] = np.sin(2 * np.pi * flight_data_test['depart_minute'] / 1440)
flight_data_test['depart_minute_cos'] = np.cos(2 * np.pi * flight_data_test['depart_minute'] / 1440)

flight_data_train.drop(columns=['scheduledoffblocktime', 'depart_month', 'depart_day', 'depart_dayofweek', 'depart_minute'], axis=1, inplace=True)
flight_data_test.drop(columns=['scheduledoffblocktime', 'depart_month', 'depart_day', 'depart_dayofweek', 'depart_minute'], axis=1, inplace=True)

# Split data into features and labels
X_train = flight_data_train.drop(columns=['delay_in_secs', 'finalflightstatus'], axis=1)
X_test = flight_data_test.drop(columns=['delay_in_secs', 'finalflightstatus'], axis=1)

y_train = flight_data_train['finalflightstatus']
y_test = flight_data_test['finalflightstatus']

# Encode labels
y_train = y_train.map({'On-Time': 0, 'Delayed': 1})
y_test = y_test.map({'On-Time': 0, 'Delayed': 1})

# High cardinality encoding
high_cardinality_cols = ['airlinecode_iata', 'destination_iata', 'aircraft_iata', 'publicgatenumber']
catboost_encoder = CatBoostEncoder(cols=high_cardinality_cols, return_df=True)
X_train_encoded = catboost_encoder.fit_transform(X_train, y_train)
X_test_encoded = catboost_encoder.transform(X_test)
X_train = X_train_encoded
X_test = X_test_encoded

# One-hot encoding
one_hot_column = ['skyc1', 'skyc2', 'traffictypecode', 'aircraftterminal']
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoded = ohe.fit_transform(X_train[one_hot_column])
ohe_new_columns = ohe.get_feature_names_out(one_hot_column)
encoded_df = pd.DataFrame(encoded, columns=ohe_new_columns)
X_train = pd.concat([X_train.drop(columns=one_hot_column), encoded_df], axis=1)
encoded = ohe.transform(X_test[one_hot_column])
ohe_new_columns = ohe.get_feature_names_out(one_hot_column)
encoded_df = pd.DataFrame(encoded, columns=ohe_new_columns)
X_test = pd.concat([X_test.drop(columns=one_hot_column), encoded_df], axis=1)

# Scale numerical columns
numerical_cols = ['tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'alti', 'vsby', 'skyl1', 'skyl2']
scaler = MinMaxScaler(feature_range=(0, 1))
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Apply HDBSCAN clustering
hdbscan_model = HDBSCAN(
    min_cluster_size=3000,
    min_samples=5,
    cluster_selection_epsilon=0.5,
    cluster_selection_method='eom',
    prediction_data=True,
    core_dist_n_jobs=-1
)
clusters_train = pd.DataFrame(hdbscan_model.fit_predict(X_train_balanced), columns=['Cluster'])
X_train_balanced['cluster'] = clusters_train['Cluster']

# Train a preclassifier (Random Forest) to predict clusters
param_space = {
    'n_estimators': Integer(50, 200),
    'max_depth': Integer(5, 50),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 5),
    'max_features': Categorical(['sqrt', 'log2']),
    'criterion': Categorical(['gini', 'entropy'])
}

bayes_cv = BayesSearchCV(
    estimator=RandomForestClassifier(),
    search_spaces=param_space,
    n_iter=5,
    cv=3,
    n_jobs=-1,
    scoring='accuracy',
    random_state=42
)
bayes_cv.fit(X_train_balanced[X_train_balanced['cluster'] != -1].drop(columns=['cluster']), X_train_balanced['cluster'][X_train_balanced['cluster'] != -1])
print(bayes_cv.best_score_)
pre_classifier = bayes_cv.best_estimator_
pre_classifier_probabilities = pre_classifier.predict_proba(X_test.drop(columns=['cluster']))

# Train a DNN model for each cluster
cluster_models = {}
model_weights_f1 = {}
model_weights_cluster = {}
entropy_weights = {}
cluster_centers = {}

for cluster in X_train_balanced['cluster'].unique():
    if cluster == -1:  # Skip noise points
        continue
    
    print(f"Training model for Cluster {cluster}")
    # Subset the training data for the current cluster
    X_cluster = X_train_balanced[X_train_balanced['cluster'] == cluster].drop(columns=['cluster']).values
    y_cluster = y_train_balanced[X_train_balanced['cluster'] == cluster].values
    
    # Create Dataset and DataLoader
    train_dataset = FlightDataset(X_cluster, y_cluster)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Define model, criterion, and optimizer
    input_dim = X_cluster.shape[1]
    model = FlightDelayDNN(input_dim=input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    train_model(model, train_loader, criterion, optimizer, epochs=10)
    
    # Save the trained model for the cluster
    cluster_models[cluster] = model
    
    # Cluster weighted
    cluster_weight = len(y_cluster) / len(y_train_balanced)
    model_weights_cluster[cluster] = cluster_weight
    
    # Calculate f1 weighted value (placeholder for actual f1 score calculation)
    model_weights_f1[cluster] = 1.0  # Placeholder for actual model evaluation score
    
    # Calculate entropy weighted value (placeholder for actual entropy calculation)
    entropy_weights[cluster] = 1.0  # Placeholder for actual entropy calculation
    
    # Calculate cluster center
    cluster_centers[cluster] = np.mean(X_cluster, axis=0)

# Normalize entropy weights
total_entropy_weight = sum(entropy_weights.values())
for model_cluster in entropy_weights:
    entropy_weights[model_cluster] /= total_entropy_weight

# Make predictions on test set using trained DNNs
model.eval()
clusters_test = pd.DataFrame(approximate_predict(hdbscan_model, X_test)[0], columns=['Cluster'])
X_test['cluster'] = clusters_test['Cluster']

test_dataset = FlightDataset(X_test.drop(columns=['cluster']).values, y_test.values)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
all_y_true = []
final_predictions_f1_weighted = []
final_predictions_cluster_weighted = []
final_predictions_entropy_weighted = []
final_predictions_non_weighted = []
final_predictions_pre_classifier = []

with torch.no_grad():
    for idx, (X_batch, y_batch) in enumerate(test_loader):
        cluster = X_test.iloc[idx]['cluster']
        if cluster == -1:
            # Find nearest cluster using euclidean distance to cluster centroids
            X_point = X_test.drop(columns=['cluster']).iloc[idx]
            min_dist = float('inf')
            nearest_cluster = None
            for c in cluster_centers.keys():
                centroid = cluster_centers[c]
                dist = np.linalg.norm(X_point - centroid)
                if dist < min_dist:
                    min_dist = dist
                    nearest_cluster = c
            cluster = nearest_cluster
        
        model = cluster_models.get(cluster)
        if model:
            outputs = model(X_batch).squeeze()
            predictions = (outputs >= 0.5).int()
            final_predictions_non_weighted.append(predictions.item())
            all_y_true.extend(y_batch.tolist())
            
            # Weighted predictions
            final_predictions_f1_weighted.append(predictions.item() * model_weights_f1[cluster])
            final_predictions_cluster_weighted.append(predictions.item() * model_weights_cluster[cluster])
            final_predictions_entropy_weighted.append(predictions.item() * entropy_weights[cluster])
            final_predictions_pre_classifier.append(np.argmax(pre_classifier_probabilities[idx]))

# Calculate overall metrics
##########################
# Weighted Average F1 Ensemble
##########################
overall_accuracy_f1 = accuracy_score(all_y_true, np.round(final_predictions_f1_weighted))
overall_precision_f1 = precision_score(all_y_true, np.round(final_predictions_f1_weighted))
overall_recall_f1 = recall_score(all_y_true, np.round(final_predictions_f1_weighted))
overall_f1_score_f1 = f1_score(all_y_true, np.round(final_predictions_f1_weighted))
print("\nOverall Metrics (Weighted Average F1 Ensemble):")
print(f"Accuracy: {overall_accuracy_f1:.4f}")
print(f"Precision: {overall_precision_f1:.4f}")
print(f"Recall: {overall_recall_f1:.4f}")
print(f"F1 Score: {overall_f1_score_f1:.4f}")

##########################
# Weighted Average Cluster Size Ensemble
##########################
overall_accuracy_cluster = accuracy_score(all_y_true, np.round(final_predictions_cluster_weighted))
overall_precision_cluster = precision_score(all_y_true, np.round(final_predictions_cluster_weighted))
overall_recall_cluster = recall_score(all_y_true, np.round(final_predictions_cluster_weighted))
overall_f1_score_cluster = f1_score(all_y_true, np.round(final_predictions_cluster_weighted))
print("\nOverall Metrics (Weighted Average Cluster Size Ensemble):")
print(f"Accuracy: {overall_accuracy_cluster:.4f}")
print(f"Precision: {overall_precision_cluster:.4f}")
print(f"Recall: {overall_recall_cluster:.4f}")
print(f"F1 Score: {overall_f1_score_cluster:.4f}")

##########################
# Entropy Weighted Ensemble
##########################
overall_accuracy_entropy = accuracy_score(all_y_true, np.round(final_predictions_entropy_weighted))
overall_precision_entropy = precision_score(all_y_true, np.round(final_predictions_entropy_weighted))
overall_recall_entropy = recall_score(all_y_true, np.round(final_predictions_entropy_weighted))
overall_f1_score_entropy = f1_score(all_y_true, np.round(final_predictions_entropy_weighted))
print("\nOverall Metrics (Entropy Weighted Ensemble):")
print(f"Accuracy: {overall_accuracy_entropy:.4f}")
print(f"Precision: {overall_precision_entropy:.4f}")
print(f"Recall: {overall_recall_entropy:.4f}")
print(f"F1 Score: {overall_f1_score_entropy:.4f}")

##########################
# Non-Weighted Ensemble
##########################
overall_accuracy_non_weighted = accuracy_score(all_y_true, final_predictions_non_weighted)
overall_precision_non_weighted = precision_score(all_y_true, final_predictions_non_weighted)
overall_recall_non_weighted = recall_score(all_y_true, final_predictions_non_weighted)
overall_f1_score_non_weighted = f1_score(all_y_true, final_predictions_non_weighted)
print("\nOverall Metrics (Non-Weighted Ensemble):")
print(f"Accuracy: {overall_accuracy_non_weighted:.4f}")
print(f"Precision: {overall_precision_non_weighted:.4f}")
print(f"Recall: {overall_recall_non_weighted:.4f}")
print(f"F1 Score: {overall_f1_score_non_weighted:.4f}")

##########################
# Pre-Classifier
##########################
overall_accuracy_pre_classifier = accuracy_score(all_y_true, final_predictions_pre_classifier)
overall_precision_pre_classifier = precision_score(all_y_true, final_predictions_pre_classifier)
overall_recall_pre_classifier = recall_score(all_y_true, final_predictions_pre_classifier)
overall_f1_score_pre_classifier = f1_score(all_y_true, final_predictions_pre_classifier)
print("\nOverall Metrics (Pre-Classifier):")
print(f"Accuracy: {overall_accuracy_pre_classifier:.4f}")
print(f"Precision: {overall_precision_pre_classifier:.4f}")
print(f"Recall: {overall_recall_pre_classifier:.4f}")
print(f"F1 Score: {overall_f1_score_pre_classifier:.4f}")
