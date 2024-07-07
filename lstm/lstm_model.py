import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.relu(out)
        return out

# Custom Dataset class to load preprocessed data
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Main training function
def train_lstm_model(train_loader, val_loader, input_size, hidden_size, num_layers, num_classes, num_epochs=10, learning_rate=0.001):
    # Initialize model
    model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for i, (features, labels) in enumerate(train_loader):
            features = features.unsqueeze(2).to(device)  # Add additional dimension for LSTM
            labels = labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for features, labels in val_loader:
                features = features.unsqueeze(2).to(device)
                labels = labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            accuracy = correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'lstm_classifier.pth')

# Example usage
if __name__ == "__main__":
    # Paths
    train_path = 'train_dataset.csv'
    val_path = 'val_dataset.csv'

    # Load preprocessed data
    X_train = np.loadtxt(train_path, delimiter=',')
    X_val = np.loadtxt(val_path, delimiter=',')

    y_train = X_train[:, -1].astype(int)
    y_val = X_val[:, -1].astype(int)

    X_train = X_train[:, :-1]
    X_val = X_val[:, :-1]

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Create DataLoader instances
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

    # Define hyperparameters
    input_size = X_train.shape[1]  # Number of features (adjust based on your data)
    hidden_size = 128
    num_layers = 2
    num_classes = 2
    num_epochs = 10
    learning_rate = 0.001

    # Train the model
    train_lstm_model(train_loader, val_loader, input_size, hidden_size, num_layers, num_classes, num_epochs, learning_rate)
