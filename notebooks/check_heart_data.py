import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

data = pd.read_csv(BASE_DIR / 'data' / 'heart.csv')

print(data.head())
print("\nColumns:")
print(data.columns)
print("\nShape:", data.shape)
