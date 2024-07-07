import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

import os

def load_data(filename):
    # Open the CSV file
    data = pd.read_csv(filename)

    # Extract features and labels
    features = data.iloc[:, :-1].values  # All columns except the last one are features
    labels = data.iloc[:, -1].values     # The last column is the label

    return features, labels

# Load data from both CSV files
X_train, y_train = [], []
csv_folder = 'C:/lstm/csvs' 
for filename in ["tree.csv", "three.csv"]:
    # Join the folder path and file name
    file_path = os.path.join(csv_folder, filename)
    features, labels = load_data(file_path)
    X_train.append(features)
    y_train.append(labels)

# Combine data and labels from both classes
X_train = np.vstack(X_train)
y_train = np.concatenate(y_train)

# Encode labels as integers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# One-hot encode labels
y_train_onehot = to_categorical(y_train_encoded)

# Save the combined data and labels if needed
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train_onehot)

print("Data loading complete.")

