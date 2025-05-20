
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and normalize the data
X = np.load("X.npy")  # Shape: (N, 1000, 12)
y = np.load("y.npy")  # Shape: (N, 9000, 12)
y = y[:, :1000, :]

# Normalize features across all samples and timesteps
X_mean = X.mean(axis=(0, 1), keepdims=True)
X_std = X.std(axis=(0, 1), keepdims=True)
X = (X - X_mean) / (X_std + 1e-8)

y_mean = y.mean(axis=(0, 1), keepdims=True)
y_std = y.std(axis=(0, 1), keepdims=True)
y = (y - y_mean) / (y_std + 1e-8)

# 2. Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 3. Dataset and DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# 4. Define LSTM model with Dropout
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMPredictor, self).__init__()
        self.gru = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, input_size)  # output per time step is 12
        #self.output_seq_len = output_seq_len

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out)  # (batch, seq_len, 12)


# 5. Instantiate model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMPredictor().to(device)
#weights = torch.tensor([770, 45, 1.05])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 6. Training loop
for epoch in range(1, 101):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                outputs = model(val_X)
                val_loss += criterion(outputs, val_y).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch}, Training Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

model.eval()
total_mse = total_mae = 0
r2_scores = []

with torch.no_grad():
    for val_X, val_y in val_loader:
        val_X, val_y = val_X.to(device), val_y.to(device)
        outputs = model(val_X)

        total_mse += nn.functional.mse_loss(outputs, val_y, reduction='sum').item()
        total_mae += nn.functional.l1_loss(outputs, val_y, reduction='sum').item()

        # Optionally compute R^2 score
        pred_np = outputs.cpu().numpy().reshape(-1, 12)
        true_np = val_y.cpu().numpy().reshape(-1, 12)
        r2_scores.append(r2_score(true_np, pred_np))

avg_mse = total_mse / len(val_dataset)
avg_mae = total_mae / len(val_dataset)
avg_r2 = np.mean(r2_scores)

print(f"\nValidation MSE: {avg_mse:.4f}")
print(f"Validation MAE: {avg_mae:.4f}")
print(f"Validation R² Score: {avg_r2:.4f}")

import matplotlib.pyplot as plt

# Hämta en sekvens från valideringsdatan
val_X_sample, val_y_sample = val_dataset[0]  # eller vilken index du vill
val_X_sample = val_X_sample.unsqueeze(0).to(device)  # lägg till batch-dimension

# Gör prediktion
model.eval()
with torch.no_grad():
    pred_y_sample = model(val_X_sample).squeeze(0).cpu().numpy()  # (seq_len, 12)

# Avnormalisera både ground truth och prediction
true_y_sample = val_y_sample.numpy()
true_y_sample = true_y_sample * (y_std.squeeze()) + y_mean.squeeze()
pred_y_sample = pred_y_sample * (y_std.squeeze()) + y_mean.squeeze()

# Extrahera t.ex. x- och y-koordinat för kropp 0
true_x = true_y_sample[:, 0]
true_y = true_y_sample[:, 1]

pred_x = pred_y_sample[:, 0]
pred_y = pred_y_sample[:, 1]
print(len(pred_y))
print(len(true_y))
# Plotta
plt.figure(figsize=(8, 6))
plt.plot(true_x, true_y, label="Ground truth", linewidth=2)
plt.plot(pred_x, pred_y, label="Prediction", linestyle='dashed')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectory prediction for body 0")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()

