import pandas as pd
import os

def split_csv_data(file_path, train_path, val_path, split_ratio=0.8):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path)
    
    # Shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split the data into training and validation sets
    train_size = int(len(data) * split_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    # Save the training and validation sets to separate CSV files
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    
    # Print the number of records in each set
    print(f"Number of records in the training set: {len(train_data)}")
    print(f"Number of records in the validation set: {len(val_data)}")

# Define file paths
file_path = 'C:/lstm/dataset.csv'
train_path = 'C:/lstm/train_dataset.csv'
val_path = 'C:/lstm/val_dataset.csv'

# Call the function to split the data
split_csv_data(file_path, train_path, val_path)
print("Data split into training and validation sets.")
