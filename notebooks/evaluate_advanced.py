"""
Comprehensive Evaluation Suite for Health Risk Prediction
==========================================================
Generates a professional evaluation report comparing old vs new models:
  - ROC Curves (per base model + ensemble)
  - Precision-Recall Curves
  - Confusion Matrix Heatmaps
  - Calibration Plots
  - Feature Importance (permutation-based)
  - Cross-Validation Stability Box Plot
  - Old vs New Comparison Table
  
All plots saved as PNG in results/ directory.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_auc_score,
    accuracy_score, f1_score, recall_score, precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'model'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
COLORS = {'xgb': '#E74C3C', 'rf': '#27AE60', 'gb': '#3498DB', 'stacking': '#8E44AD', 'old': '#95A5A6'}


def evaluate_old_model(dataset_name, data, target_col, model_type='logistic'):
    """Reproduce the OLD model's metrics for fair comparison."""
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=300, max_depth=8,
                                        min_samples_split=5, class_weight='balanced',
                                        random_state=42)
    
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'brier': brier_score_loss(y_test, y_prob),
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
    }


def evaluate_new_model(dataset_name, data, target_col, use_smote=False):
    """Evaluate the NEW stacking ensemble model."""
    prefix = 'diabetes' if 'diabetes' in dataset_name.lower() else 'heart'
    
    model = joblib.load(MODEL_DIR / f'{prefix}_stacking_model.pkl')
    scaler = joblib.load(MODEL_DIR / f'{prefix}_advanced_scaler.pkl')
    feature_names = joblib.load(MODEL_DIR / f'{prefix}_feature_names.pkl')
    
    X = data[feature_names]
    y = data[target_col]
    
    # Use the same split for fair comparison
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    X_test_s = scaler.transform(X_test)
    
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    
    # Also get per-base-model predictions
    X_train_s = scaler.transform(X_train)
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_s, y_train = smote.fit_resample(X_train_s, y_train)
    
    base_model_probs = {}
    for name, estimator in model.named_estimators_.items():
        try:
            base_model_probs[name] = estimator.predict_proba(X_test_s)[:, 1]
        except Exception:
            base_model_probs[name] = estimator.decision_function(X_test_s)
    
    # Cross-validation scores
    X_scaled_all = scaler.transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled_all, data[target_col], cv=cv, scoring='roc_auc')
    
    # Permutation importance
    perm_imp = permutation_importance(model, X_test_s, y_test, n_repeats=10,
                                       random_state=42, scoring='roc_auc')
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'brier': brier_score_loss(y_test, y_prob),
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'base_model_probs': base_model_probs,
        'cv_scores': cv_scores,
        'perm_importance': perm_imp,
        'feature_names': feature_names,
        'model': model,
        'X_test_s': X_test_s,
    }


def plot_roc_curves(old_results, new_results, dataset_name):
    """ROC curves for old model, each base model, and stacking ensemble."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Old model
    fpr, tpr, _ = roc_curve(old_results['y_test'], old_results['y_prob'])
    ax.plot(fpr, tpr, color=COLORS['old'], linewidth=2, linestyle='--',
            label=f"Old Model (AUC = {old_results['roc_auc']:.4f})")
    
    # Base models
    color_map = {'xgb': COLORS['xgb'], 'rf': COLORS['rf'], 'gb': COLORS['gb']}
    name_map = {'xgb': 'XGBoost', 'rf': 'Random Forest', 'gb': 'Gradient Boosting'}
    
    for name, probs in new_results['base_model_probs'].items():
        try:
            fpr_b, tpr_b, _ = roc_curve(new_results['y_test'], probs)
            auc_val = auc(fpr_b, tpr_b)
            ax.plot(fpr_b, tpr_b, color=color_map.get(name, 'gray'), linewidth=1.5,
                    alpha=0.7, label=f"{name_map.get(name, name)} (AUC = {auc_val:.4f})")
        except Exception:
            pass
    
    # Stacking ensemble
    fpr_s, tpr_s, _ = roc_curve(new_results['y_test'], new_results['y_prob'])
    ax.plot(fpr_s, tpr_s, color=COLORS['stacking'], linewidth=3,
            label=f"Stacking Ensemble (AUC = {new_results['roc_auc']:.4f})")
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title(f'{dataset_name} — ROC Curves: Old vs New', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{dataset_name.lower().replace(" ", "_")}_roc_curves.png', dpi=150)
    plt.close()
    print(f"  ✓ ROC curves saved")


def plot_precision_recall(new_results, dataset_name):
    """Precision-Recall curve (more meaningful for imbalanced data)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    precision, recall, _ = precision_recall_curve(new_results['y_test'], new_results['y_prob'])
    ap = average_precision_score(new_results['y_test'], new_results['y_prob'])
    
    ax.plot(recall, precision, color=COLORS['stacking'], linewidth=2.5,
            label=f'Stacking Ensemble (AP = {ap:.4f})')
    
    for name, probs in new_results['base_model_probs'].items():
        try:
            p, r, _ = precision_recall_curve(new_results['y_test'], probs)
            ap_b = average_precision_score(new_results['y_test'], probs)
            color_map = {'xgb': COLORS['xgb'], 'rf': COLORS['rf'], 'gb': COLORS['gb']}
            name_map = {'xgb': 'XGBoost', 'rf': 'Random Forest', 'gb': 'Gradient Boosting'}
            ax.plot(r, p, color=color_map.get(name, 'gray'), linewidth=1.5, alpha=0.7,
                    label=f"{name_map.get(name, name)} (AP = {ap_b:.4f})")
        except Exception:
            pass
    
    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title(f'{dataset_name} — Precision-Recall Curve', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{dataset_name.lower().replace(" ", "_")}_pr_curve.png', dpi=150)
    plt.close()
    print(f"  ✓ Precision-Recall curve saved")


def plot_confusion_matrix(new_results, dataset_name):
    """Annotated confusion matrix heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    cm = confusion_matrix(new_results['y_test'], new_results['y_pred'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'],
                annot_kws={'size': 16})
    
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('Actual', fontsize=13)
    ax.set_title(f'{dataset_name} — Confusion Matrix (Stacking Ensemble)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{dataset_name.lower().replace(" ", "_")}_confusion_matrix.png', dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrix saved")


def plot_calibration(new_results, dataset_name):
    """Calibration plot: are predicted probabilities well-calibrated?"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    prob_true, prob_pred = calibration_curve(
        new_results['y_test'], new_results['y_prob'], n_bins=8, strategy='uniform'
    )
    
    ax.plot(prob_pred, prob_true, 's-', color=COLORS['stacking'], linewidth=2,
            markersize=8, label='Stacking Ensemble')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfectly Calibrated')
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=13)
    ax.set_ylabel('Fraction of Positives', fontsize=13)
    ax.set_title(f'{dataset_name} — Calibration Plot', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    
    brier = new_results['brier']
    ax.text(0.05, 0.9, f'Brier Score: {brier:.4f}\n(lower is better)',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{dataset_name.lower().replace(" ", "_")}_calibration.png', dpi=150)
    plt.close()
    print(f"  ✓ Calibration plot saved")


def plot_feature_importance(new_results, dataset_name, top_n=15):
    """Permutation-based feature importance (top N)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    perm_imp = new_results['perm_importance']
    feature_names = new_results['feature_names']
    
    imp_mean = perm_imp.importances_mean
    imp_std = perm_imp.importances_std
    indices = np.argsort(imp_mean)[::-1][:top_n]
    
    names = [feature_names[i] for i in indices]
    values = imp_mean[indices]
    errors = imp_std[indices]
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))
    
    ax.barh(range(len(names)), values, xerr=errors, color=colors,
            edgecolor='white', linewidth=0.5, capsize=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Decrease in ROC-AUC', fontsize=13)
    ax.set_title(f'{dataset_name} — Feature Importance (Permutation)', fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{dataset_name.lower().replace(" ", "_")}_feature_importance.png', dpi=150)
    plt.close()
    print(f"  ✓ Feature importance saved")


def plot_cv_boxplot(diabetes_cv, heart_cv):
    """Box plot comparing CV score distributions."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    data_plot = [diabetes_cv, heart_cv]
    bp = ax.boxplot(data_plot, labels=['Diabetes', 'Heart Disease'],
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='white', markersize=8))
    
    colors_box = [COLORS['xgb'], COLORS['gb']]
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('ROC-AUC Score', fontsize=13)
    ax.set_title('Cross-Validation Score Distribution', fontsize=15, fontweight='bold')
    ax.set_ylim([0.7, 1.0])
    
    for i, cv in enumerate(data_plot):
        ax.text(i + 1, cv.mean() + 0.015, f'{cv.mean():.4f}',
                ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'cv_score_distribution.png', dpi=150)
    plt.close()
    print(f"  ✓ CV box plot saved")


def create_comparison_table(d_old, d_new, h_old, h_new):
    """Print and save old vs new comparison table."""
    
    header = f"\n{'='*80}\n{'EVALUATION RESULTS — OLD vs NEW':^80}\n{'='*80}\n"
    
    table_header = f"\n{'Metric':<25} {'Diabetes OLD':<15} {'Diabetes NEW':<15} {'Heart OLD':<15} {'Heart NEW':<15}"
    separator = "-" * 85
    
    metrics_list = [
        ('ROC-AUC', 'roc_auc'),
        ('Accuracy', 'accuracy'),
        ('F1-Score', 'f1'),
        ('Precision', 'precision'),
        ('Recall (Sensitivity)', 'recall'),
        ('Brier Score ↓', 'brier'),
    ]
    
    lines = [header, table_header, separator]
    
    for metric_name, metric_key in metrics_list:
        line = f"{metric_name:<25} {d_old[metric_key]:<15.4f} {d_new[metric_key]:<15.4f} {h_old[metric_key]:<15.4f} {h_new[metric_key]:<15.4f}"
        lines.append(line)
    
    lines.append(separator)
    
    # Improvements
    lines.append("\nIMPROVEMENTS:")
    d_roc_imp = (d_new['roc_auc'] - d_old['roc_auc']) * 100
    h_roc_imp = (h_new['roc_auc'] - h_old['roc_auc']) * 100
    lines.append(f"  Diabetes ROC-AUC:       +{d_roc_imp:.2f} percentage points")
    lines.append(f"  Heart Disease ROC-AUC:  +{h_roc_imp:.2f} percentage points")
    
    d_f1_imp = (d_new['f1'] - d_old['f1']) * 100
    h_f1_imp = (h_new['f1'] - h_old['f1']) * 100
    lines.append(f"  Diabetes F1-Score:      +{d_f1_imp:.2f} percentage points")
    lines.append(f"  Heart Disease F1-Score: +{h_f1_imp:.2f} percentage points")
    
    result = "\n".join(lines)
    print(result)
    
    # Save to file
    with open(RESULTS_DIR / 'comparison_table.txt', 'w', encoding='utf-8') as f:
        f.write(result)
    print(f"\n  ✓ Comparison table saved to results/comparison_table.txt")
    
    return result


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("COMPREHENSIVE EVALUATION SUITE")
    print("=" * 60)
    
    # --- Load original datasets (for old model evaluation) ---
    diabetes_orig = pd.read_csv(BASE_DIR / 'data' / 'diabetes.csv')
    diabetes_orig.columns = diabetes_orig.columns.str.strip().str.lower()
    zero_cols = ['glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi']
    for col in zero_cols:
        if col in diabetes_orig.columns:
            diabetes_orig[col] = diabetes_orig[col].replace(0, np.nan)
    diabetes_orig.fillna(diabetes_orig.mean(), inplace=True)
    
    heart_orig = pd.read_csv(BASE_DIR / 'data' / 'heart.csv')
    heart_orig.columns = heart_orig.columns.str.strip().str.lower()
    heart_orig.replace('?', np.nan, inplace=True)
    for col in heart_orig.columns:
        heart_orig[col] = pd.to_numeric(heart_orig[col], errors='coerce')
    heart_orig.fillna(heart_orig.mean(), inplace=True)
    heart_orig['diagnosis'] = heart_orig['diagnosis'].apply(lambda x: 1 if x > 0 else 0)
    
    # --- Load featured datasets (for new model evaluation) ---
    diabetes_feat = pd.read_csv(BASE_DIR / 'data' / 'diabetes_featured.csv')
    heart_feat = pd.read_csv(BASE_DIR / 'data' / 'heart_featured.csv')
    
    # ============================================================
    # EVALUATE OLD MODELS
    # ============================================================
    print("\n--- Evaluating OLD models ---")
    d_old = evaluate_old_model('Diabetes', diabetes_orig, 'outcome', 'logistic')
    print(f"  Diabetes (old):     ROC-AUC = {d_old['roc_auc']:.4f}")
    
    h_old = evaluate_old_model('Heart Disease', heart_orig, 'diagnosis', 'random_forest')
    print(f"  Heart Disease (old): ROC-AUC = {h_old['roc_auc']:.4f}")
    
    # ============================================================
    # EVALUATE NEW MODELS
    # ============================================================
    print("\n--- Evaluating NEW stacking ensemble models ---")
    d_new = evaluate_new_model('Diabetes', diabetes_feat, 'outcome', use_smote=True)
    print(f"  Diabetes (new):     ROC-AUC = {d_new['roc_auc']:.4f}")
    
    h_new = evaluate_new_model('Heart Disease', heart_feat, 'diagnosis', use_smote=False)
    print(f"  Heart Disease (new): ROC-AUC = {h_new['roc_auc']:.4f}")
    
    # ============================================================
    # GENERATE PLOTS
    # ============================================================
    print("\n--- Generating evaluation plots ---")
    
    plot_roc_curves(d_old, d_new, 'Diabetes')
    plot_roc_curves(h_old, h_new, 'Heart Disease')
    
    plot_precision_recall(d_new, 'Diabetes')
    plot_precision_recall(h_new, 'Heart Disease')
    
    plot_confusion_matrix(d_new, 'Diabetes')
    plot_confusion_matrix(h_new, 'Heart Disease')
    
    plot_calibration(d_new, 'Diabetes')
    plot_calibration(h_new, 'Heart Disease')
    
    plot_feature_importance(d_new, 'Diabetes')
    plot_feature_importance(h_new, 'Heart Disease')
    
    plot_cv_boxplot(d_new['cv_scores'], h_new['cv_scores'])
    
    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    create_comparison_table(d_old, d_new, h_old, h_new)
    
    # ============================================================
    # DETAILED CLASSIFICATION REPORTS
    # ============================================================
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORTS")
    print("=" * 60)
    
    print("\n--- Diabetes (Stacking Ensemble) ---")
    print(classification_report(d_new['y_test'], d_new['y_pred'],
                                target_names=['No Diabetes', 'Diabetes']))
    
    print("\n--- Heart Disease (Stacking Ensemble) ---")
    print(classification_report(h_new['y_test'], h_new['y_pred'],
                                target_names=['No Disease', 'Heart Disease']))
    
    # Cross-validation summary
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Diabetes:     {d_new['cv_scores'].mean():.4f} ± {d_new['cv_scores'].std():.4f}")
    print(f"Heart Disease: {h_new['cv_scores'].mean():.4f} ± {h_new['cv_scores'].std():.4f}")
    
    print(f"\n✓ All plots saved to: {RESULTS_DIR}")
    print("✓ Evaluation complete!")
