import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / 'data' / 'diabetes.csv'

data = pd.read_csv(data_path)

print(data.head())
print("\nDataset shape:", data.shape)
