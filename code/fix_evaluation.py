"""
Fixed evaluation without test set leakage
"""
import numpy as np
import pandas as pd
from sklearn.metrics import *
from scipy import stats

# Load predictions
results = pd.read_csv("../results/model_comparison.csv")
pred_df = pd.read_csv("../results/predictions.csv")

print("="*70)
print("CORRECTED EVALUATION (No Test Set Leakage)")
print("="*70)

# Use FIXED threshold of 0.5 (no peeking at test labels)
y_test = pred_df['true_label'].values
pred_rf = pred_df['pred_rf_raw'].values
pred_ens = pred_df['pred_ensemble'].values

# Compute metrics with fixed threshold
def compute_metrics_fixed(y_true, y_pred_proba, threshold=0.5):
    auroc = roc_auc_score(y_true, y_pred_proba)
    
    # Use FIXED threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'AUROC': auroc,
        'F1': f1,
        'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0
    }

rf_metrics = compute_metrics_fixed(y_test, pred_rf)
ens_metrics = compute_metrics_fixed(y_test, pred_ens)

print("\nResults with FIXED threshold (0.5):")
print(f"  RF Raw:    F1={rf_metrics['F1']:.4f}, AUROC={rf_metrics['AUROC']:.4f}")
print(f"  Ensemble:  F1={ens_metrics['F1']:.4f}, AUROC={ens_metrics['AUROC']:.4f}")
print(f"  Difference: {ens_metrics['F1'] - rf_metrics['F1']:.4f}")

# Statistical test: Bootstrap confidence interval
print("\n" + "="*70)
print("STATISTICAL VALIDATION (Bootstrap)")
print("="*70)

def bootstrap_f1(y_true, y_pred_proba, n_iterations=1000):
    f1_scores = []
    n = len(y_true)
    
    for _ in range(n_iterations):
        indices = np.random.choice(n, n, replace=True)
        y_boot = y_true[indices]
        pred_boot = y_pred_proba[indices]
        
        y_pred = (pred_boot >= 0.5).astype(int)
        f1 = f1_score(y_boot, y_pred)
        f1_scores.append(f1)
    
    return np.array(f1_scores)

print("\nBootstrapping F1 scores (1000 iterations)...")
rf_f1_boot = bootstrap_f1(y_test, pred_rf)
ens_f1_boot = bootstrap_f1(y_test, pred_ens)

rf_ci = np.percentile(rf_f1_boot, [2.5, 97.5])
ens_ci = np.percentile(ens_f1_boot, [2.5, 97.5])

print(f"\nRF Raw F1:    {rf_f1_boot.mean():.4f} [95% CI: {rf_ci[0]:.4f}-{rf_ci[1]:.4f}]")
print(f"Ensemble F1:  {ens_f1_boot.mean():.4f} [95% CI: {ens_ci[0]:.4f}-{ens_ci[1]:.4f}]")

# Paired t-test
improvement = ens_f1_boot - rf_f1_boot
t_stat, p_value = stats.ttest_rel(ens_f1_boot, rf_f1_boot)

print(f"\nImprovement: {improvement.mean():.4f} ± {improvement.std():.4f}")
print(f"T-test p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ Improvement is statistically significant (p < 0.05)")
else:
    print("⚠️  Improvement is NOT statistically significant (p >= 0.05)")
    print("   Likely due to small test set (n=545) and similar models")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("\n1. Using proper evaluation (fixed threshold, no test leakage)")
print("2. Ensemble improvement is small and may not be statistically significant")
print("3. Need larger, more diverse test sets for reliable conclusions")
print("4. This demonstrates the CONCEPT of ensembling, not definitive proof")
