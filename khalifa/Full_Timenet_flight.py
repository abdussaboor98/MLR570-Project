import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import time  # For timing
from sklearn.metrics import mean_absolute_error, mean_squared_error  # For metrics

# Load the dataset
file_path = 'flight_data_full.xlsx'
df = pd.read_excel(file_path)

# Ensure `publicscheduleddatetime` is in datetime format
df['publicscheduleddatetime'] = pd.to_datetime(df['publicscheduleddatetime'])

# Sort the data by `publicscheduleddatetime`
df = df.sort_values(by='publicscheduleddatetime').reset_index(drop=True)

# Define the target column
target_column = 'actual_difference_seconds'

# Select features for the model
features = df.drop(columns=['publicscheduleddatetime', target_column])
target = df[target_column]

# Sequence length
seq_len = 24

############# WHAT IS THIS DOING ###############
# Generate sequences
X, y = [], []
for i in range(len(features) - seq_len):
    X.append(features.iloc[i:i+seq_len].values)
    y.append(target.iloc[i+seq_len])
###############################################

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the dataset without shuffling
split_index = int(0.9 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Create PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# GPU Support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# TimesBlock and TimesNet Definitions (as provided above)
class TimesBlock(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size=64, num_periods=2):
        super(TimesBlock, self).__init__()
        self.num_periods = num_periods
        self.hidden_size = hidden_size

        self.conv_intra = nn.Conv2d(1, hidden_size, kernel_size=(1, num_periods), stride=1)
        self.conv_inter = nn.Conv2d(hidden_size, hidden_size, kernel_size=(num_periods, 1), stride=1)
        self.relu = nn.ReLU()

        self._dummy_input = torch.zeros((1, 1, seq_len, input_size))
        self.flattened_size = self._calculate_flattened_size()

        self.fc = nn.Linear(self.flattened_size, seq_len * input_size)

    def _calculate_flattened_size(self):
        with torch.no_grad():
            x_intra = self.relu(self.conv_intra(self._dummy_input))
            x_inter = self.relu(self.conv_inter(x_intra))
            return x_inter.numel()

    def forward(self, x):
        x_2d = rearrange(x, 'b s f -> b 1 s f')
        x_intra = self.relu(self.conv_intra(x_2d))
        x_inter = self.relu(self.conv_inter(x_intra))
        x_flatten = torch.flatten(x_inter, start_dim=1)
        x_out = self.fc(x_flatten)
        x_out = x_out.view(x.size(0), x.size(1), x.size(2))
        return x_out

class TimesNet(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size=64, num_blocks=2):
        super(TimesNet, self).__init__()
        self.blocks = nn.ModuleList([TimesBlock(input_size, seq_len, hidden_size) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(input_size, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.fc_out(x.mean(dim=1))
        return x

# Initialize the model
input_size = X_train.shape[2]
model = TimesNet(input_size=input_size, seq_len=seq_len).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with timing
epochs = 10
start_time = time.time()  # Start timing
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    epoch_start = time.time()  # Epoch timing
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to GPU
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    epoch_end = time.time()  # End epoch timing
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss/len(train_loader)}, Time: {epoch_end - epoch_start:.2f} seconds")

# Calculate total training time
total_time = time.time() - start_time
print(f"Total training time: {total_time:.2f} seconds")

# Evaluate the model
model.eval()
test_loss = 0.0
y_true, y_pred = [], []  # To store true and predicted values for metrics

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to GPU
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        test_loss += loss.item()
        y_true.extend(y_batch.cpu().numpy())
        y_pred.extend(outputs.cpu().numpy())

# Calculate metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)

print(f"Test Loss (MSE): {test_loss/len(test_loader)}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")

