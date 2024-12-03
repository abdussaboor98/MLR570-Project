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
X_train, y_train = smote.fit_resample(X_train, y_train)
print(f'After SMOTE: X_train: {X_train.shape} y_train: {y_train.shape}')


X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long).to(device)
X_val_tensor = torch.tensor(X_val.to_numpy(), dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long).to(device)

# Create DataLoader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


print(f'X_train_tensor.shape: {X_train_tensor.shape}')

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
model_path = 'flight_delay_autoencoder.pth'
input_size = X_train.shape[1]
middle_layer_sizes = [512, 512, 256, 256, 128, 128, 64]  # Specify the sizes of the middle layers
latent_size = 64  # Set your desired feature size here
dropout_rates = [0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3]  # Specify dropout rates for each middle layer
batch_norm_layers=[True, False, True, False, True, True, True]
model = FlightDelayAutoEncoder(input_size, middle_layer_sizes, latent_size=latent_size, dropout_rates=dropout_rates, batch_norm_layers=batch_norm_layers).to(device)
criterion = nn.MSELoss() 
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
scaler = GradScaler('cuda')

# Check if the model file exists
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print('Model loaded from', model_path)
else:
    print('No pre-trained Autoencoder model found. Training from scratch.')
    num_epochs = 1000
    patience = 10
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
    plt.show()

    # Save the model
    torch.save(model.state_dict(), model_path)
    print('Model saved as', model_path)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

print("Training MLP with AE")
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 64, 128, 32),
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    learning_rate="adaptive",
    
    random_state=42,
    verbose=True,
)
mlp.fit(train_features, y_train)
y_pred_mlp = mlp.predict(test_features)
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
mlp_precision = precision_score(y_test, y_pred_mlp)
mlp_recall = recall_score(y_test, y_pred_mlp)
mlp_f1 = f1_score(y_test, y_pred_mlp)
print(f"MLP Accuracy: {mlp_accuracy:.4f}")
print(f"MLP Precision: {mlp_precision:.4f}")
print(f"MLP Recall: {mlp_recall:.4f}")
print(f"MLP F1 Score: {mlp_f1:.4f}")

print("Training MLP without AE")
mlp_no_se = MLPClassifier(
    hidden_layer_sizes=(256, 64, 128, 32),
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    learning_rate="adaptive",
    random_state=42,
    verbose=True,
)
mlp_no_se.fit(X_train, y_train)
y_pred_mlp_no_se = mlp_no_se.predict(X_test)
mlp_no_se_accuracy = accuracy_score(y_test, y_pred_mlp_no_se)
mlp_no_se_precision = precision_score(y_test, y_pred_mlp_no_se)
mlp_no_se_recall = recall_score(y_test, y_pred_mlp_no_se)
mlp_no_se_f1 = f1_score(y_test, y_pred_mlp_no_se)
print(f"MLP Accuracy: {mlp_no_se_accuracy:.4f}")
print(f"MLP Precision: {mlp_no_se_precision:.4f}")
print(f"MLP Recall: {mlp_no_se_recall:.4f}")
print(f"MLP F1 Score: {mlp_no_se_f1:.4f}")

# XGBoost Classifier
# print('Training XGBoost')
# param_space_xgb = {
#     'n_estimators': Integer(50, 200),
#     'gamma': Real(1e-6, 1e-1, prior='log-uniform'),
#     'max_depth': Integer(2, 64),
#     'subsample': Real(0.4, 1.0, prior='uniform'),
#     'reg_lambda': Real(1e-3, 1e1, prior='log-uniform'),
#     'learning_rate': Real(1e-7, 1e-2, prior='log-uniform')
# }

# bayes_cv = BayesSearchCV(
#     estimator=XGBClassifier(booster='gbtree'),
#     search_spaces=param_space_xgb,
#     n_iter=1,
#     cv=5,
#     n_jobs=1,
#     scoring='f1',
#     random_state=42
# )
# bayes_cv.fit(train_features, y_train)
# best_xgb_f = bayes_cv.best_estimator_
# y_pred_xgb = best_xgb_f.predict(test_features)
# xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
# xgb_precision = precision_score(y_test, y_pred_xgb)
# xgb_recall = recall_score(y_test, y_pred_xgb)
# xgb_f1 = f1_score(y_test, y_pred_xgb)
# print('XGBoost using latent space')
# print(f'XGBoost Accuracy: {xgb_accuracy:.4f}')
# print(f'XGBoost Precision: {xgb_precision:.4f}')
# print(f'XGBoost Recall: {xgb_recall:.4f}')
# print(f'XGBoost F1 Score: {xgb_f1:.4f}')


# bayes_cv.fit(X_train, y_train)
# best_xgb = bayes_cv.best_estimator_
# y_pred_xgb = best_xgb.predict(test_features)
# xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
# xgb_precision = precision_score(y_test, y_pred_xgb)
# xgb_recall = recall_score(y_test, y_pred_xgb)
# xgb_f1 = f1_score(y_test, y_pred_xgb)
# print('XGBoost using X_train')
# print(f'XGBoost Accuracy: {xgb_accuracy:.4f}')
# print(f'XGBoost Precision: {xgb_precision:.4f}')
# print(f'XGBoost Recall: {xgb_recall:.4f}')
# print(f'XGBoost F1 Score: {xgb_f1:.4f}')


# Train downstream ML models using the extracted features
# Random Forest Classifier
# rf = RandomForestClassifier()
# rf.fit(train_features, y_train)
# y_pred_rf = rf.predict(test_features)
# rf_accuracy = accuracy_score(y_test, y_pred_rf)
# rf_precision = precision_score(y_test, y_pred_rf)
# rf_recall = recall_score(y_test, y_pred_rf)
# rf_f1 = f1_score(y_test, y_pred_rf)
# print(f'Random Forest Accuracy: {rf_accuracy:.4f}')
# print(f'Random Forest Precision: {rf_precision:.4f}')
# print(f'Random Forest Recall: {rf_recall:.4f}')
# print(f'Random Forest F1 Score: {rf_f1:.4f}')

# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# y_pred_rf = rf.predict(X_test)
# rf_accuracy = accuracy_score(y_test, y_pred_rf)
# rf_precision = precision_score(y_test, y_pred_rf)
# rf_recall = recall_score(y_test, y_pred_rf)
# rf_f1 = f1_score(y_test, y_pred_rf)
# print(f'Random Forest Accuracy: {rf_accuracy:.4f}')
# print(f'Random Forest Precision: {rf_precision:.4f}')
# print(f'Random Forest Recall: {rf_recall:.4f}')
# print(f'Random Forest F1 Score: {rf_f1:.4f}')
