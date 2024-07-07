import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def pre_process_data(X):
    # Normalize features (assuming features are within a range)
    scaler = MinMaxScaler(feature_range=(0, 1))  # Scales between 0 and 1
    X = scaler.fit_transform(X)

    # Save the scaler for future use
    joblib.dump(scaler, 'scaler.pkl')

    return X

# Load the combined data
X = np.load('X_train.npy', allow_pickle=True)

# Preprocess the data
X_preprocessed = pre_process_data(X)

# Save the preprocessed data
np.save('X_preprocessed.npy', X_preprocessed)

print("Data preprocessing complete.")
