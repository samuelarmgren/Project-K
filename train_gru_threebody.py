
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and normalize the data
X = np.load("X.npy")  # Shape: (N, 10, 12)
y = np.load("y.npy")  # Shape: (N,)

print(y)

# Normalize features across all samples and timesteps
X_mean = X.mean(axis=(0, 1), keepdims=True)
X_std = X.std(axis=(0, 1), keepdims=True)
X = (X - X_mean) / (X_std + 1e-8)

# 2. Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# 3. Dataset and DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# 4. Define GRU model with Dropout
class GRUClassifier(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=1, dropout=0.3, num_classes=3):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # use output from the last time step
        return self.fc(out)

# 5. Instantiate model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 6. Training loop
for epoch in range(1, 1001):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0 or epoch == 1:
        model.eval()
        correct = total = 0
        all_preds = [0, 1, 2]
        all_labels = [0, 1, 2]

        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                outputs = model(val_X)
                _, predicted = torch.max(outputs.data, 1)
                total += val_y.size(0)
                correct += (predicted == val_y).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(val_y.cpu().numpy())
        acc = correct / total
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val Accuracy: {acc:.4f}")

# 7. Confusion matrix
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Convergent", "Stable", "Divergent"]))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Convergent", "Stable", "Divergent"],
            yticklabels=["Convergent", "Stable", "Divergent"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# 8. Save the model
torch.save(model.state_dict(), "gru_model.pth")
print("Model saved to gru_model.pth")
