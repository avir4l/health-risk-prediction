import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Locate project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load dataset
data = pd.read_csv(BASE_DIR / 'data' / 'heart.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# Replace '?' with NaN
data.replace('?', np.nan, inplace=True)

# Convert all columns to numeric
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill missing values
data.fillna(data.mean(), inplace=True)

# Target column
target_col = 'diagnosis'

X = data.drop(target_col, axis=1)
y = data[target_col].apply(lambda x: 1 if x > 0 else 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest (NO scaling needed)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Heart Disease RF Accuracy:", accuracy)

# Save model
joblib.dump(model, BASE_DIR / 'model' / 'heart_model.pkl')
