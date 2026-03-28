"""
Domain-Specific Feature Engineering for Health Risk Prediction
===============================================================
Creates medically-motivated features that capture known clinical
relationships, giving the models richer signal than raw values alone.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# 1. DIABETES FEATURE ENGINEERING
# ============================================================
print("=" * 60)
print("FEATURE ENGINEERING — DIABETES DATASET")
print("=" * 60)

diabetes = pd.read_csv(BASE_DIR / 'data' / 'diabetes_advanced_clean.csv')
print(f"Original features: {list(diabetes.columns)}")
print(f"Original shape: {diabetes.shape}")

# --- Interaction features ---
# Glucose × BMI: insulin resistance correlates with both
diabetes['glucose_bmi_interaction'] = diabetes['glucose'] * diabetes['bmi']

# Age × Glucose: older patients with high glucose = compounded risk
diabetes['age_glucose_risk'] = diabetes['age'] * diabetes['glucose'] / 1000

# BMI × Blood Pressure: obesity-hypertension synergy
diabetes['bmi_bp_interaction'] = diabetes['bmi'] * diabetes['bloodpressure']

# --- Ratio features ---
# Insulin resistance proxy (simplified HOMA-IR)
diabetes['insulin_resistance_proxy'] = diabetes['glucose'] / (diabetes['insulin'] + 1)

# Glucose-to-age ratio (age-adjusted glucose)
diabetes['glucose_age_ratio'] = diabetes['glucose'] / (diabetes['age'] + 1)

# --- Composite risk scores ---
# Metabolic syndrome score: count of how many risk factors are elevated
diabetes['metabolic_syndrome_score'] = (
    (diabetes['glucose'] > 100).astype(int) +
    (diabetes['bmi'] > 25).astype(int) +
    (diabetes['bloodpressure'] > 130).astype(int) +
    (diabetes['insulin'] > 166).astype(int)   # median insulin in diabetics
)

# --- Binned features (non-linear effects) ---
diabetes['age_group'] = pd.cut(
    diabetes['age'],
    bins=[0, 30, 45, 60, 100],
    labels=[0, 1, 2, 3]
).astype(int)

diabetes['bmi_category'] = pd.cut(
    diabetes['bmi'],
    bins=[0, 18.5, 25, 30, 100],
    labels=[0, 1, 2, 3]
).astype(int)

diabetes['glucose_category'] = pd.cut(
    diabetes['glucose'],
    bins=[0, 100, 126, 600],
    labels=[0, 1, 2]
).astype(int)

# --- Polynomial features for key predictors ---
diabetes['glucose_squared'] = diabetes['glucose'] ** 2
diabetes['bmi_squared'] = diabetes['bmi'] ** 2

# Save
diabetes_fe_path = BASE_DIR / 'data' / 'diabetes_featured.csv'
diabetes.to_csv(diabetes_fe_path, index=False)
print(f"\nNew features added: {diabetes.shape[1] - 9} new columns")
print(f"Final shape: {diabetes.shape}")
print(f"✓ Saved to {diabetes_fe_path}")

# ============================================================
# 2. HEART DISEASE FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING — HEART DISEASE DATASET")
print("=" * 60)

heart = pd.read_csv(BASE_DIR / 'data' / 'heart_advanced_clean.csv')
print(f"Original features: {list(heart.columns)}")
print(f"Original shape: {heart.shape}")

# --- Interaction features ---
# Age × Blood Pressure: combined age-hypertension risk
heart['age_bp_risk'] = heart['age'] * heart['trestbps'] / 1000

# Heart rate reserve: how far from predicted max HR
heart['heart_rate_reserve'] = (220 - heart['age']) - heart['thalach']

# ST depression severity: combines magnitude and slope
heart['st_severity'] = heart['oldpeak'] * heart['slope']

# Age × Cholesterol interaction
heart['age_chol_interaction'] = heart['age'] * heart['chol'] / 1000

# --- Ratio features ---
# Age-adjusted cholesterol
heart['chol_age_ratio'] = heart['chol'] / (heart['age'] + 1)

# Thalach as percentage of predicted max HR
heart['hr_percent_max'] = heart['thalach'] / (220 - heart['age'] + 1) * 100

# --- Composite risk scores ---
# Cardiac risk score (Framingham-style composite)
heart['cardiac_risk_score'] = (
    (heart['age'] > 55).astype(int) +
    (heart['trestbps'] > 130).astype(int) +
    (heart['chol'] > 240).astype(int) +
    (heart['fbs'] == 1).astype(int) +
    (heart['thalach'] < 120).astype(int) +
    (heart['exang'] == 1).astype(int) +
    (heart['oldpeak'] > 1.5).astype(int) +
    (heart['ca'] > 0).astype(int)
)

# Symptom severity score
heart['symptom_severity'] = (
    (heart['cp'].isin([1, 2])).astype(int) * 2 +   # angina types
    (heart['exang'] == 1).astype(int) * 2 +
    (heart['oldpeak'] > 2).astype(int)
)

# --- Binned features ---
heart['age_group'] = pd.cut(
    heart['age'],
    bins=[0, 40, 55, 65, 100],
    labels=[0, 1, 2, 3]
).astype(int)

heart['bp_category'] = pd.cut(
    heart['trestbps'],
    bins=[0, 120, 130, 140, 300],
    labels=[0, 1, 2, 3]
).astype(int)

# --- Polynomial features ---
heart['oldpeak_squared'] = heart['oldpeak'] ** 2
heart['age_squared'] = heart['age'] ** 2

# Save
heart_fe_path = BASE_DIR / 'data' / 'heart_featured.csv'
heart.to_csv(heart_fe_path, index=False)
print(f"\nNew features added: {heart.shape[1] - 14} new columns")
print(f"Final shape: {heart.shape}")
print(f"✓ Saved to {heart_fe_path}")

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)
