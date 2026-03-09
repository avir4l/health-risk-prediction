import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Load dataset
data = pd.read_csv(BASE_DIR / 'data' / 'heart.csv')

# Clean column names
data.columns = data.columns.str.strip()

# Replace '?' with NaN
data.replace('?', np.nan, inplace=True)

# Convert all columns to numeric
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Convert diagnosis to binary target
data['target'] = data['diagnosis'].apply(lambda x: 0 if x == 0 else 1)

# Drop old diagnosis column
data.drop('diagnosis', axis=1, inplace=True)

# Fill missing values with mean
data.fillna(data.mean(), inplace=True)

# 🔴 SAVE CLEANED DATASET
clean_path = BASE_DIR / 'data' / 'heart_cleaned.csv'
data.to_csv(clean_path, index=False)

print("Heart dataset cleaned and saved")
print("Saved at:", clean_path)
print("\nMissing values check:")
print(data.isnull().sum())
