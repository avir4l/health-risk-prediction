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

# Load diabetes dataset
data = pd.read_csv(BASE_DIR / 'data' / 'diabetes.csv')

# 🔑 FIX: Normalize column names
data.columns = data.columns.str.strip().str.lower()

# Replace zero values with NaN (medically invalid)
cols_with_zero = [
    'glucose',
    'bloodpressure',
    'bmi',
    'insulin',
    'skinthickness'
]

# Apply replacement only if column exists
for col in cols_with_zero:
    if col in data.columns:
        data[col] = data[col].replace(0, np.nan)

# Fill missing values
data.fillna(data.mean(), inplace=True)

# Features and target
X = data.drop('outcome', axis=1)
y = data['outcome']

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
