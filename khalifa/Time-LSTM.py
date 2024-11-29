import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
file_path = r'C:\Users\Admin\Documents\Processed_TIMELSTM.csv'
df = pd.read_csv(file_path)

# Ensure the columns align correctly
features = df.drop(columns=['actual_difference_seconds'])  # Exclude the target column
target = df['actual_difference_seconds']  # Define the target column

# Sequence generation (fixed rows with irregular intervals)
seq_len = 24  # Number of rows per sequence
X, y = [], []
for i in range(len(features) - seq_len):
    X.append(features.iloc[i:i+seq_len].values)  # Include all features, including time_diff
    y.append(target.iloc[i+seq_len])  # Target is the next value after the sequence

# Convert to numpy arrays
X = np.array(X)  # Shape: (num_sequences, seq_len, num_features)
y = np.array(y)  # Shape: (num_sequences,)

# Train-test split
split_index = int(0.9 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Dataset class for PyTorch
class TimeLSTMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeLSTMDataset(X_train, y_train)
test_dataset = TimeLSTMDataset(X_test, y_test)

# DataLoader for batching
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Time-LSTM Model
class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.lstm = nn.LSTMCell(input_size - 1, hidden_size)  # Exclude time_diff
        self.time_gate = nn.Linear(1, hidden_size)  # For time differences
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)

        for t in range(seq_len):
            time_diff = x[:, t, 0].unsqueeze(1)  # Extract time_diff (assume it's the first feature)
            inputs = x[:, t, 1:]  # Exclude time_diff
            time_decay = torch.sigmoid(self.time_gate(time_diff))  # Time gate
            c_t = c_t * time_decay  # Apply time decay to cell state
            h_t, c_t = self.lstm(inputs, (h_t, c_t))  # Update LSTM state

        out = self.fc(h_t)  # Predict from final hidden state
        return out

# Dynamically determine input_size
input_size = X.shape[2]
print("Input size for TimeLSTM:", input_size)

# Initialize model, loss function, optimizer
hidden_size = 64
model = TimeLSTM(input_size, hidden_size).to('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)  # Pass sequences through Time-LSTM
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader)}")

# Evaluation
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        y_true.extend(y_batch.cpu().numpy())
        y_pred.extend(outputs.squeeze().cpu().numpy())

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
print(f"Test MAE: {mae}, Test MSE: {mse}")

