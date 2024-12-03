import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical, Real
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from sklearn.neural_network import MLPClassifier
from hdbscan import HDBSCAN

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

print('Encoding')
# High cardinality columns - CatBoostEncoder
high_cardinality_cols = ['airlinecode_iata', 'destination_iata', 'aircraft_iata', 'publicgatenumber']

# One-hot encoding
one_hot_column =  ['skyc1', 'skyc2', 'traffictypecode', 'aircraftterminal', 'wxcodes'] + high_cardinality_cols
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoded = ohe.fit_transform(X_train[one_hot_column])
ohe_new_columns = ohe.get_feature_names_out(one_hot_column)
encoded_df = pd.DataFrame(encoded, columns=ohe_new_columns)
X_train = pd.concat([X_train.drop(columns=one_hot_column), encoded_df], axis=1)
encoded = ohe.transform(X_test[one_hot_column])
encoded_df = pd.DataFrame(encoded, columns=ohe_new_columns)
X_test = pd.concat([X_test.drop(columns=one_hot_column), encoded_df], axis=1)

# StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

train_size = int(0.9 * len(X_train))
X_train, X_val = X_train[:train_size], X_train[train_size:]
y_train, y_val = y_train[:train_size], y_train[train_size:]

print(f'X_train: {X_train.shape} X_val: {X_val.shape}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using: {device}')


smote = SMOTE()
print(f'Before SMOTE: X_train: {X_train.shape} y_train: {y_train.shape}')
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f'After SMOTE: X_train: {X_train_balanced.shape} y_train: {y_train_balanced.shape}')


X_train_balanced_tensor = torch.tensor(X_train_balanced.to_numpy(), dtype=torch.float32).to(device)
y_train_balanced_tensor = torch.tensor(y_train_balanced.to_numpy(), dtype=torch.long).to(device)
X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long).to(device)
X_val_tensor = torch.tensor(X_val.to_numpy(), dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long).to(device)

# Create DataLoader for training
train_balanced_dataset = TensorDataset(X_train_balanced_tensor, y_train_balanced_tensor)
train_loader = DataLoader(train_balanced_dataset, batch_size=64, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


print(f'X_train_tensor.shape: {X_train_balanced_tensor.shape}')

class FlightDelayAutoEncoder(nn.Module):
    def __init__(self, input_size, middle_layer_sizes, latent_size=50, dropout_rates=[0.5], batch_norm_layers=None , batch_norm_encoder=True, batch_norm_decoder=True):
        super(FlightDelayAutoEncoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        layer_sizes = [input_size] + middle_layer_sizes + [latent_size]
        if batch_norm_layers is None:
            batch_norm_layers = [True] * (len(layer_sizes) - 1)
        for i in range(len(layer_sizes) - 1):  # Encoder construction with optional BatchNorm  # Encoder construction
            self.encoder.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2 and batch_norm_layers[i]:
                self.encoder.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                self.encoder.append(nn.ReLU())
                dropout_rate = dropout_rates[i] if i < len(dropout_rates) else dropout_rates[-1]
                self.encoder.append(nn.Dropout(dropout_rate))
        for i in range(len(layer_sizes) - 2, -1, -1):  # Decoder construction with optional BatchNorm
            self.decoder.append(nn.Linear(layer_sizes[i + 1], layer_sizes[i]))
            if i > 0 and batch_norm_layers[i - 1]:
                if batch_norm_decoder:
                    self.decoder.append(nn.BatchNorm1d(layer_sizes[i]))
                self.decoder.append(nn.ReLU())
                dropout_rate = dropout_rates[i - 1] if i - 1 < len(dropout_rates) else dropout_rates[-1]
                self.decoder.append(nn.Dropout(dropout_rate))
        

    def forward(self, x):  # Forward pass for autoencoder
        for layer in self.encoder:
            x = layer(x)
        latent = x
        for layer in self.decoder:
            x = layer(x)
        output = x  # Extracted features before sigmoid
        return latent, output


import os

# Define the path to save the model
model_name = 'flight_delay_autoencoder_50_3layers'
model_path = f'./{model_name}.pth'
input_size = X_train.shape[1]
middle_layer_sizes = [512, 256, 128]  # Specify the sizes of the middle layers
latent_size = 50  # Set your desired feature size here
dropout_rates = [0.2, 0.2, 0.2]  # Specify dropout rates for each middle layer
batch_norm_layers=[True, True, True]
model = FlightDelayAutoEncoder(input_size, middle_layer_sizes, latent_size=latent_size, dropout_rates=dropout_rates, batch_norm_layers=batch_norm_layers).to(device)
criterion = nn.MSELoss() 
optimizer = optim.AdamW(model.parameters(), lr=1e-6)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
scaler = GradScaler('cuda')

# Check if the model file exists
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print('Model loaded from', model_path)
else:
    print('No pre-trained Autoencoder model found. Training from scratch.')
    num_epochs = 1000
    patience = 20
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_state = None
    model.train()
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        for i, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            with autocast('cuda'):
                features, outputs = model(X_batch)
                outputs = outputs.squeeze()
                loss = criterion(outputs, X_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_losses.append(loss.item())

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                _, val_outputs = model(X_val_batch)
                val_outputs = val_outputs.squeeze()
                val_loss += criterion(val_outputs, X_val_batch).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

        if best_model_state == None:
            best_model_state = model.state_dict()

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            model.load_state_dict(best_model_state)
            break

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(model_name + '_loss.png')

    # Save the model
    torch.save(model.state_dict(), model_path)
    print('Model saved as', model_path)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print('Extracting features')
train_features_list = []
test_features_list = []

model.eval()
with torch.no_grad():
    for X_batch, _ in train_loader:
        features, _ = model(X_batch)
        train_features_list.append(features.cpu())
    for X_batch, _ in test_loader:
        features, _ = model(X_batch)
        test_features_list.append(features.cpu())

train_features = torch.cat(train_features_list, dim=0)
test_features = torch.cat(test_features_list, dim=0)

print('Converting to numpy')
train_features = train_features.numpy()
test_features = test_features.numpy()

print('Training HDBSCAN')
hdbscan_model = HDBSCAN(
    min_cluster_size=5000,      # Increase to avoid microclusters
    min_samples=10,              # Lower to reduce noise points
    cluster_selection_epsilon=0.5,  # Increase to reduce noise points
    cluster_selection_method='leaf',  # 'eom' tends to produce more balanced clusters
    prediction_data=True,
    core_dist_n_jobs=-1
)
clusters_train = pd.DataFrame(hdbscan_model.fit_predict(X_train), columns=['Cluster'])
print(clusters_train['Cluster'].value_counts())
