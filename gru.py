import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Simulate random dataset (replace this with real data)
def generate_fake_data(num_samples=1000):
    
    X = np.load("X.npy")   # Shape: (num_samples, 10, 12)
    y = np.load("y.npy")   # Shape: (num_samples,)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# Define GRU model for classification
class GRUClassifier(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=1, num_classes=3):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)  # out: (batch, seq_len, hidden)
        out = out[:, -1, :]   # Use last time step's output
        out = self.fc(out)
        return out

# Prepare dataset
X, y = generate_fake_data(1000)
train_X, val_X = X[:800], X[800:]
train_y, val_y = y[:800], y[800:]

# Initialize model
model = GRUClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    output = model(train_X)
    loss = criterion(output, train_y)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        val_preds = model(val_X).argmax(dim=1)
        acc = (val_preds == val_y).float().mean().item()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Accuracy: {acc:.4f}")

# Inference example
test_sample = X[0].unsqueeze(0)  # shape: (1, 10, 12)
predicted_class = model(test_sample).argmax(dim=1).item()
print(f"Predicted class: {predicted_class} (0=conv, 1=stable, 2=div)")
