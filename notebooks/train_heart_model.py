import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

# Fill missing values with column mean
data.fillna(data.mean(), inplace=True)

# Target column
target_col = 'diagnosis'

# Features and target
X = data.drop(target_col, axis=1)
y = data[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Heart Disease Model Accuracy:", accuracy)

# Save model and scaler
joblib.dump(model, BASE_DIR / 'model' / 'heart_model.pkl')
joblib.dump(scaler, BASE_DIR / 'model' / 'heart_scaler.pkl')
