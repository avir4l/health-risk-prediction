"""
Advanced Preprocessing Pipeline for Health Risk Prediction
===========================================================
Replaces naive mean imputation with:
  - KNN Imputation (uses feature correlations)
  - IQR-based outlier capping (preserves rows)
  - Saves cleaned datasets for downstream use
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer

BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# 1. DIABETES DATASET
# ============================================================
print("=" * 60)
print("PREPROCESSING DIABETES DATASET")
print("=" * 60)

diabetes = pd.read_csv(BASE_DIR / 'data' / 'diabetes.csv')
diabetes.columns = diabetes.columns.str.strip().str.lower()

print(f"Original shape: {diabetes.shape}")
print(f"Original zeros in key columns:")

# Columns where 0 is medically invalid
zero_invalid_cols = ['glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi']

for col in zero_invalid_cols:
    if col in diabetes.columns:
        n_zeros = (diabetes[col] == 0).sum()
        print(f"  {col}: {n_zeros} zeros ({n_zeros/len(diabetes)*100:.1f}%)")
        diabetes[col] = diabetes[col].replace(0, np.nan)

# KNN Imputation (k=5, uses 5 nearest neighbors to estimate missing values)
print("\nApplying KNN Imputation (k=5)...")
feature_cols = [c for c in diabetes.columns if c != 'outcome']
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
diabetes[feature_cols] = knn_imputer.fit_transform(diabetes[feature_cols])

print(f"Missing values after KNN imputation: {diabetes[feature_cols].isnull().sum().sum()}")

# IQR-based outlier capping (clip, don't remove)
print("\nApplying IQR outlier capping...")
for col in feature_cols:
    Q1 = diabetes[col].quantile(0.25)
    Q3 = diabetes[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((diabetes[col] < lower) | (diabetes[col] > upper)).sum()
    if n_outliers > 0:
        print(f"  {col}: {n_outliers} outliers capped")
        diabetes[col] = diabetes[col].clip(lower, upper)

# Save
diabetes_clean_path = BASE_DIR / 'data' / 'diabetes_advanced_clean.csv'
diabetes.to_csv(diabetes_clean_path, index=False)
print(f"\n✓ Saved cleaned diabetes dataset to {diabetes_clean_path}")
print(f"  Shape: {diabetes.shape}")

# ============================================================
# 2. HEART DISEASE DATASET
# ============================================================
print("\n" + "=" * 60)
print("PREPROCESSING HEART DISEASE DATASET")
print("=" * 60)

heart = pd.read_csv(BASE_DIR / 'data' / 'heart.csv')
heart.columns = heart.columns.str.strip().str.lower()

print(f"Original shape: {heart.shape}")

# Replace '?' with NaN
heart.replace('?', np.nan, inplace=True)

# Convert to numeric
for col in heart.columns:
    heart[col] = pd.to_numeric(heart[col], errors='coerce')

n_missing = heart.isnull().sum().sum()
print(f"Total missing values (from '?'): {n_missing}")

# Convert diagnosis to binary (0 = no disease, 1+ = disease)
heart['diagnosis'] = heart['diagnosis'].apply(lambda x: 1 if x > 0 else 0)

# KNN Imputation
print("\nApplying KNN Imputation (k=5)...")
feature_cols_h = [c for c in heart.columns if c != 'diagnosis']
knn_imputer_h = KNNImputer(n_neighbors=5, weights='distance')
heart[feature_cols_h] = knn_imputer_h.fit_transform(heart[feature_cols_h])

print(f"Missing values after KNN imputation: {heart[feature_cols_h].isnull().sum().sum()}")

# IQR-based outlier capping
print("\nApplying IQR outlier capping...")
continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for col in continuous_cols:
    if col in heart.columns:
        Q1 = heart[col].quantile(0.25)
        Q3 = heart[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((heart[col] < lower) | (heart[col] > upper)).sum()
        if n_outliers > 0:
            print(f"  {col}: {n_outliers} outliers capped")
            heart[col] = heart[col].clip(lower, upper)

# Save
heart_clean_path = BASE_DIR / 'data' / 'heart_advanced_clean.csv'
heart.to_csv(heart_clean_path, index=False)
print(f"\n✓ Saved cleaned heart dataset to {heart_clean_path}")
print(f"  Shape: {heart.shape}")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
