import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from lstm_model import LSTM, CustomDataset  # Import your LSTM model and CustomDataset class

# Function to calculate accuracy
def calculate_accuracy(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.unsqueeze(2).to(device)
            labels = labels.to(device)

            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    # Paths and configurations (adjust as needed)
    model_path = 'lstm_classifier.pth'  # Path to your trained model
    test_csv_file = 'test_dataset.csv'  # Path to your test CSV file
    batch_size = 32  # Batch size used during training

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load your trained model
    model = LSTM(input_size=20, hidden_size=128, num_layers=2, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create DataLoader instance for the test set
    test_dataset = CustomDataset(test_csv_file)  # Replace with your test CSV path
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Calculate accuracy on the test set
    test_accuracy = calculate_accuracy(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.4f}')
