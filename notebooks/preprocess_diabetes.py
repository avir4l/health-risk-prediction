import pandas as pd
import numpy as np
from pathlib import Path

# Locate project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load dataset
data = pd.read_csv(BASE_DIR / 'data' / 'diabetes.csv')

# Replace zero values with NaN for specific columns
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_with_zero] = data[cols_with_zero].replace(0, np.nan)

# Fill missing values with column mean
data.fillna(data.mean(), inplace=True)

print("Missing values handled")
print(data.isnull().sum())

