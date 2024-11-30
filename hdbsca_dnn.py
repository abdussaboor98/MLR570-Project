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
from sklearn.metrics import accuracy_score

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

flight_data_train.drop(columns=['scheduledoffblocktime'], axis=1, inplace=True)
flight_data_test.drop(columns=['scheduledoffblocktime'], axis=1, inplace=True)

# Split data into features and labels
X_train = flight_data_train.drop(columns=['delay_in_secs', 'finalflightstatus'], axis=1)
X_test = flight_data_test.drop(columns=['delay_in_secs', 'finalflightstatus'], axis=1)

y_train = flight_data_train['finalflightstatus']
y_test = flight_data_test['finalflightstatus']

# Encode labels
y_train = y_train.map({'On-Time': 0, 'Delayed': 1})
y_test = y_test.map({'On-Time': 0, 'Delayed': 1})

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
overall_accuracy_f1 = sum(np.array(all_y_true) == np.round(final_predictions_f1_weighted)) / len(all_y_true)
print("\nOverall Metrics (Weighted Average F1 Ensemble):")
print(f"Overall Accuracy: {overall_accuracy_f1:.4f}")

##########################
# Weighted Average Cluster Size Ensemble
##########################
overall_accuracy_cluster = sum(np.array(all_y_true) == np.round(final_predictions_cluster_weighted)) / len(all_y_true)
print("\nOverall Metrics (Weighted Average Cluster Size Ensemble):")
print(f"Overall Accuracy: {overall_accuracy_cluster:.4f}")

##########################
# Entropy Weighted Ensemble
##########################
overall_accuracy_entropy = sum(np.array(all_y_true) == np.round(final_predictions_entropy_weighted)) / len(all_y_true)
print("\nOverall Metrics (Entropy Weighted Ensemble):")
print(f"Overall Accuracy: {overall_accuracy_entropy:.4f}")

##########################
# Non-Weighted Ensemble
##########################
overall_accuracy_non_weighted = sum(np.array(all_y_true) == np.array(final_predictions_non_weighted)) / len(all_y_true)
print("\nOverall Metrics (Non-Weighted Ensemble):")
print(f"Overall Accuracy: {overall_accuracy_non_weighted:.4f}")

##########################
# Pre-Classifier
##########################
overall_accuracy_pre_classifier = sum(np.array(all_y_true) == np.array(final_predictions_pre_classifier)) / len(all_y_true)
print("\nOverall Metrics (Pre-Classifier):")
print(f"Overall Accuracy: {overall_accuracy_pre_classifier:.4f}")
