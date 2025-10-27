"""
Ensemble Analysis - Understanding Why Multiple Models Help
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

print("="*70)
print("ENSEMBLE ANALYSIS - MODEL DIVERSITY & COMPLEMENTARITY")
print("="*70)

# Load results
results = pd.read_csv("../results/model_comparison.csv")
predictions = pd.read_csv("../results/predictions.csv")

# Load full prediction details (need to recreate with all model predictions)
# For now, work with what we have

print("\n" + "="*70)
print("WHY USE ENSEMBLE?")
print("="*70)

print("\n1. VARIANCE REDUCTION")
print("   Individual models overfit to training noise")
print("   Averaging multiple models reduces variance")
print("   Formula: Var(ensemble) = Var(model) / N")

print("\n2. ERROR COMPLEMENTARITY")
print("   Different models make different mistakes")
print("   When combined, errors can cancel out")
print("   Requires models to be diverse (different algorithms/data)")

print("\n3. ROBUSTNESS")
print("   Single model can be sensitive to outliers")
print("   Ensemble is more stable across different test sets")

# ==================================================================
# ANALYSIS 1: Model Performance Comparison
# ==================================================================
print("\n" + "="*70)
print("ENSEMBLE IMPACT: Performance Comparison")
print("="*70)

print("\n" + results[['Model', 'F1', 'AUROC', 'FPR']].to_string(index=False))

# Find best individual vs best ensemble
best_individual = results[~results['Model'].str.contains('Ensemble')]['F1'].max()
best_ensemble = results[results['Model'].str.contains('Ensemble')]['F1'].max()

print(f"\n📊 Summary:")
print(f"  Best Individual Model: F1 = {best_individual:.4f}")
print(f"  Best Ensemble Model:   F1 = {best_ensemble:.4f}")
print(f"  Improvement:           {(best_ensemble - best_individual)*100:.2f}%")

if best_ensemble > best_individual:
    print(f"\n✅ Ensemble provides improvement!")
else:
    print(f"\n⚠️  Minimal improvement in this case")
    print(f"   Likely reasons:")
    print(f"     • Models are too similar (both RF and GB on same data)")
    print(f"     • Test set is small (high variance)")
    print(f"     • Individual models already near optimal")

# ==================================================================
# ANALYSIS 2: Why Ensemble Works (Theory)
# ==================================================================
print("\n" + "="*70)
print("THEORETICAL BENEFITS OF ENSEMBLE")
print("="*70)

print("\nConsider 3 models with 90% accuracy each:")
print("  • If errors are INDEPENDENT:")
print("    → Ensemble accuracy: ~97%")
print("  • If errors are CORRELATED:")
print("    → Ensemble accuracy: ~90% (no improvement)")

print("\nKey insight: Ensemble helps when models are DIVERSE")
print("  • Different algorithms (RF, GB, Neural Nets)")
print("  • Different data samples (bagging)")
print("  • Different features (random subspaces)")

print("\nIn our case:")
print("  • RF and GB are somewhat similar (both tree-based)")
print("  • Both use same features (embeddings)")
print("  • Both trained on same data")
print("  → Limited diversity → Limited improvement")

# ==================================================================
# VISUALIZATION
# ==================================================================
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: F1 Score Comparison
models = results['Model'].values
f1_scores = results['F1'].values
colors = ['steelblue' if 'Ensemble' not in m else 'coral' for m in models]

axes[0, 0].barh(range(len(models)), f1_scores, color=colors)
axes[0, 0].set_yticks(range(len(models)))
axes[0, 0].set_yticklabels(models, fontsize=9)
axes[0, 0].set_xlabel('F1 Score', fontsize=11)
axes[0, 0].set_title('Model Comparison: F1 Scores', fontsize=12, fontweight='bold')
axes[0, 0].grid(axis='x', alpha=0.3)
axes[0, 0].axvline(best_individual, color='blue', linestyle='--', alpha=0.5, label='Best Individual')
axes[0, 0].legend()

# Plot 2: AUROC Comparison
auroc_scores = results['AUROC'].values

axes[0, 1].barh(range(len(models)), auroc_scores, color=colors)
axes[0, 1].set_yticks(range(len(models)))
axes[0, 1].set_yticklabels(models, fontsize=9)
axes[0, 1].set_xlabel('AUROC', fontsize=11)
axes[0, 1].set_title('Model Comparison: AUROC', fontsize=12, fontweight='bold')
axes[0, 1].grid(axis='x', alpha=0.3)

# Plot 3: FPR Comparison (lower is better)
fpr_scores = results['FPR'].values

axes[1, 0].barh(range(len(models)), fpr_scores, color=colors)
axes[1, 0].set_yticks(range(len(models)))
axes[1, 0].set_yticklabels(models, fontsize=9)
axes[1, 0].set_xlabel('False Positive Rate (lower is better)', fontsize=11)
axes[1, 0].set_title('Model Comparison: FPR', fontsize=12, fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)

# Plot 4: Theoretical ensemble benefit
axes[1, 1].axis('off')
axes[1, 1].text(0.5, 0.9, 'Ensemble Benefits', ha='center', fontsize=14, fontweight='bold')
axes[1, 1].text(0.1, 0.7, '✓ Variance Reduction\n   (averaging smooths predictions)', fontsize=10)
axes[1, 1].text(0.1, 0.5, '✓ Error Complementarity\n   (different models, different errors)', fontsize=10)
axes[1, 1].text(0.1, 0.3, '✓ Robustness\n   (less sensitive to outliers)', fontsize=10)
axes[1, 1].text(0.1, 0.1, '⚠ Requires model diversity\n   (our models are similar)', fontsize=10, color='red')

plt.tight_layout()
plt.savefig('../plots/ensemble_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Plot saved: ensemble_analysis.png")

# ==================================================================
# RECOMMENDATIONS
# ==================================================================
print("\n" + "="*70)
print("ENSEMBLE RECOMMENDATIONS")
print("="*70)

print("\n💡 To improve ensemble performance:")
print("\n1. INCREASE DIVERSITY")
print("   • Add neural network models")
print("   • Try different embedding strategies")
print("   • Use different feature subsets")

print("\n2. ADVANCED ENSEMBLE METHODS")
print("   • Stacking: Train meta-model on predictions")
print("   • Boosting: Sequentially correct errors")
print("   • Blending: Learn optimal weights")

print("\n3. UNCERTAINTY-AWARE ENSEMBLE")
print("   • Weight by prediction confidence")
print("   • Exclude low-confidence predictions")
print("   • Report ensemble disagreement")

print("\n" + "="*70)
print("✅ ENSEMBLE ANALYSIS COMPLETE")
print("="*70)
