"""
Advanced Model Training — Stacking Ensemble
=============================================
Trains a competition-grade Stacking Ensemble for both datasets:
  Base models:  XGBoost, Random Forest, Logistic Regression
  Meta-learner: Logistic Regression

Uses:
  - SMOTE for class imbalance (diabetes)
  - Pre-tuned hyperparameters (validated via 5-fold CV)
  - Stratified 5-Fold Cross-Validation for evaluation
  - Saves models + scalers + feature lists for Flask app integration
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
)
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'model'
MODEL_DIR.mkdir(exist_ok=True)


def build_stacking_ensemble(dataset='diabetes'):
    """Build a Stacking Ensemble with pre-tuned hyperparameters."""
    
    if dataset == 'diabetes':
        xgb = XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=2.0,
            random_state=42, eval_metric='logloss',
            min_child_weight=3, gamma=0.1,
        )
        rf = RandomForestClassifier(
            n_estimators=250, max_depth=8,
            min_samples_split=4, min_samples_leaf=2,
            class_weight='balanced', max_features='sqrt',
            random_state=42,
        )
        gb = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, min_samples_split=5,
            random_state=42,
        )
    else:  # heart disease
        xgb = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.85,
            reg_alpha=0.3, reg_lambda=1.5,
            random_state=42, eval_metric='logloss',
            min_child_weight=2, gamma=0.05,
        )
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=7,
            min_samples_split=3, min_samples_leaf=2,
            class_weight='balanced', max_features='sqrt',
            random_state=42,
        )
        gb = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.85, min_samples_split=4,
            random_state=42,
        )
    
    base_models = [
        ('xgb', xgb),
        ('rf', rf),
        ('gb', gb),
    ]
    
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        cv=5,
        stack_method='predict_proba',
        passthrough=False,
        n_jobs=1,
    )
    
    return stacking


def train_model(dataset_name, data, target_col, use_smote=False):
    """Full training pipeline for one dataset."""
    
    print(f"\n{'='*60}")
    print(f"TRAINING: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    feature_names = list(X.columns)
    
    print(f"Features: {len(feature_names)}")
    print(f"Samples: {len(X)}")
    print(f"Class distribution: {dict(y.value_counts())}")
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply SMOTE if needed
    if use_smote:
        print("\nApplying SMOTE to balance classes...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        print(f"After SMOTE: {dict(pd.Series(y_resampled).value_counts())}")
    else:
        X_resampled, y_resampled = X_scaled, y
    
    # --- First, evaluate individual base models via CV ---
    prefix = 'diabetes' if 'diabetes' in dataset_name.lower() else 'heart'
    print(f"\n--- Individual base model CV scores ---")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    individual_models = {
        'XGBoost': XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.8,
            random_state=42, eval_metric='logloss'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=250, max_depth=8, class_weight='balanced',
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42
        ),
    }
    
    for name, model in individual_models.items():
        scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring='roc_auc')
        print(f"  {name:<22} ROC-AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
    
    # --- Train Stacking Ensemble ---
    print(f"\nTraining Stacking Ensemble (XGBoost + RF + GB -> LR)...")
    stacking = build_stacking_ensemble(dataset=prefix)
    stacking.fit(X_resampled, y_resampled)
    print("  Stacking model trained!")
    
    # Cross-validation scores on ORIGINAL data (not SMOTE'd) for honest eval
    print(f"\n--- Stacking Ensemble 5-Fold CV (on original data) ---")
    cv_scores = cross_val_score(stacking, X_scaled, y, cv=cv, scoring='roc_auc')
    print(f"  ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  Per fold: {[f'{s:.4f}' for s in cv_scores]}")
    
    acc_scores = cross_val_score(stacking, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"  Accuracy: {acc_scores.mean():.4f} +/- {acc_scores.std():.4f}")
    
    f1_scores = cross_val_score(stacking, X_scaled, y, cv=cv, scoring='f1')
    print(f"  F1-Score: {f1_scores.mean():.4f} +/- {f1_scores.std():.4f}")
    
    # Save model, scaler, and feature names
    joblib.dump(stacking, MODEL_DIR / f'{prefix}_stacking_model.pkl')
    joblib.dump(scaler, MODEL_DIR / f'{prefix}_advanced_scaler.pkl')
    joblib.dump(feature_names, MODEL_DIR / f'{prefix}_feature_names.pkl')
    
    print(f"\n  Saved:")
    print(f"  {prefix}_stacking_model.pkl")
    print(f"  {prefix}_advanced_scaler.pkl")
    print(f"  {prefix}_feature_names.pkl")
    
    return {
        'model': stacking,
        'scaler': scaler,
        'feature_names': feature_names,
        'cv_roc_auc_mean': cv_scores.mean(),
        'cv_roc_auc_std': cv_scores.std(),
        'cv_accuracy_mean': acc_scores.mean(),
        'cv_accuracy_std': acc_scores.std(),
        'cv_f1_mean': f1_scores.mean(),
        'cv_f1_std': f1_scores.std(),
        'cv_scores': cv_scores,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("ADVANCED MODEL TRAINING PIPELINE")
    print("Stacking Ensemble: XGBoost + Random Forest + Gradient Boosting")
    print("=" * 60)
    
    # --- Diabetes ---
    diabetes = pd.read_csv(BASE_DIR / 'data' / 'diabetes_featured.csv')
    diabetes_results = train_model(
        'Diabetes', diabetes, target_col='outcome', use_smote=True
    )
    
    # --- Heart Disease ---
    heart = pd.read_csv(BASE_DIR / 'data' / 'heart_featured.csv')
    heart_results = train_model(
        'Heart Disease', heart, target_col='diagnosis', use_smote=False
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<20} {'CV ROC-AUC':<22} {'CV Accuracy':<22} {'CV F1-Score':<20}")
    print("-" * 84)
    d = diabetes_results
    h = heart_results
    print(f"{'Diabetes':<20} {d['cv_roc_auc_mean']:.4f} +/- {d['cv_roc_auc_std']:.4f}   {d['cv_accuracy_mean']:.4f} +/- {d['cv_accuracy_std']:.4f}   {d['cv_f1_mean']:.4f} +/- {d['cv_f1_std']:.4f}")
    print(f"{'Heart Disease':<20} {h['cv_roc_auc_mean']:.4f} +/- {h['cv_roc_auc_std']:.4f}   {h['cv_accuracy_mean']:.4f} +/- {h['cv_accuracy_std']:.4f}   {h['cv_f1_mean']:.4f} +/- {h['cv_f1_std']:.4f}")
    
    print(f"\nAll models saved to: {MODEL_DIR}")
    print("Training complete!")
