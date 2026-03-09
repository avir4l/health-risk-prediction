import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    classification_report
)
from sklearn.linear_model import LogisticRegression

# Locate project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load dataset
data = pd.read_csv(BASE_DIR / 'data' / 'heart.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# Replace '?' with NaN
data.replace('?', np.nan, inplace=True)

# Convert to numeric
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill missing values
data.fillna(data.mean(), inplace=True)

# Target column
target_col = 'diagnosis'

X = data.drop(target_col, axis=1)
y = data[target_col].apply(lambda x: 1 if x > 0 else 0)

# 🔹 Increased test size to 30%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix:")
print(cm)

# Metrics
roc_auc = roc_auc_score(y_test, y_prob)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("\nROC-AUC Score:", roc_auc)
print("Sensitivity (Recall):", sensitivity)
print("Specificity:", specificity)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
