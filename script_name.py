import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

# File Path
file_path = r'C:\Users\Dell\Desktop\KSU\3_DMT\Week_filtered.csv' 

try:
    # Load Dataset
    data = pd.read_csv(file_path)

    # Display Dataset Information
    print(f"Dataset Shape: {data.shape}")
    print("\nDataset Info:")
    print(data.info())
    print("\nDataset Statistics:")
    print(data.describe())

    # Check for Missing Values
    missing_values = data.isnull().sum()
    print("\nMissing Values:\n", missing_values)

    # Analyze Class Distribution
    if 'Label' in data.columns:
        class_distribution = data['Label'].value_counts()
        print("\nClass Distribution:\n", class_distribution)
    else:
        raise KeyError("The 'Label' column is missing. Ensure the dataset contains this column.")

    # Encode Labels
    if 'Encoded_Label' not in data.columns:
        label_encoder = LabelEncoder()
        data['Encoded_Label'] = label_encoder.fit_transform(data['Label'])
        print("Encoded_Label column created successfully.")
    else:
        print("Encoded_Label column already exists.")

    # Feature Extraction
    X = data.drop(columns=['Label', 'Encoded_Label'])  # Features
    X.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaN
    X.fillna(X.mean(), inplace=True)  # Replace NaN with column mean
    print("Features extracted successfully!")

    # Display Cleaned Data Sample
    print("\nCleaned Data Sample:\n", X.head())

    # Optional: Save the Processed Data for Debugging
    processed_file_path = 'processed_data.csv'
    data.to_csv(processed_file_path, index=False)
    print(f"Processed data saved as '{processed_file_path}' for review.")

except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Ensure the file is in the correct path.")
except KeyError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
