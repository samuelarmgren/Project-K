import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class ThreeBodyDataset(torch.utils.data.Dataset):
    def __init__(self, X_path='X.npy', y_path='y.npy'):
        self.X = np.load(X_path)  # shape: (N, 10, 12)
        self.y = np.load(y_path)  # shape: (N,)
        self.X = self.X.reshape(self.X.shape[0], -1) # creates (N, 120)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(120, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 output classes
        )

    def forward(self, x):
        return self.model(x)

def train(model, dataloader, criterion, optimizer, epochs=100):
    for epoch in range(1, epochs+1):
        total_loss = 0
        correct = 0
        total = 0

        model.train()
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

# Evaluation function for confusion matrix and classification report
def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Convergent", "Stable", "Divergent"]))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Convergent", "Stable", "Divergent"],
                yticklabels=["Convergent", "Stable", "Divergent"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix_mlp_test.png")
    plt.close()

# Saving the model
def save_model(model, filename="mlp_model.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

# Load data
dataset = ThreeBodyDataset("X.npy", "y.npy")
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Split into train and test datasets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, loss, optimizer
model = MLPClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(model, train_dataloader, criterion, optimizer, epochs=100)

# Evaluate the model
evaluate(model, test_dataloader)

# Save the model after training
save_model(model, "mlp_model.pth")

# Test on a sample
model.eval()
with torch.no_grad():
    sample = torch.tensor(dataset.X[0], dtype=torch.float32).unsqueeze(0)
    output = model(sample)
    predicted_class = torch.argmax(output, dim=1).item()
    print("Predicted class:", ["Convergent", "Stable", "Divergent"][predicted_class])
