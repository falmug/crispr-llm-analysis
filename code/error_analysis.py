"""
Error Analysis - Understanding Model Failures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("ERROR ANALYSIS - MODEL FAILURES")
print("="*70)

# Load predictions
pred_df = pd.read_csv("../results/predictions.csv")

# Add error flag
pred_df['correct_rf'] = ((pred_df['pred_rf_raw'] >= 0.5).astype(int) == pred_df['true_label']).astype(int)
pred_df['correct_ensemble'] = ((pred_df['pred_ensemble'] >= 0.5).astype(int) == pred_df['true_label']).astype(int)

# Overall accuracy
rf_acc = pred_df['correct_rf'].mean()
ens_acc = pred_df['correct_ensemble'].mean()

print(f"\nOverall Performance:")
print(f"  RF Raw Accuracy: {rf_acc*100:.1f}%")
print(f"  Ensemble Accuracy: {ens_acc*100:.1f}%")
print(f"  Improvement: {(ens_acc-rf_acc)*100:.1f}%")

# ==================================================================
# ERROR PATTERN 1: By Cell Line
# ==================================================================
print("\n" + "="*70)
print("ERROR PATTERN #1: Performance by Cell Line")
print("="*70)

cell_perf = pred_df.groupby('cell').agg({
    'correct_rf': 'mean',
    'correct_ensemble': 'mean',
    'gene': 'count'
}).rename(columns={'gene': 'count'})

print("\n" + cell_perf.to_string())

# Which cell line is harder?
if cell_perf['correct_rf'].iloc[0] < cell_perf['correct_rf'].iloc[1]:
    harder_cell = cell_perf.index[0]
    easier_cell = cell_perf.index[1]
else:
    harder_cell = cell_perf.index[1]
    easier_cell = cell_perf.index[0]

print(f"\n⚠️  Finding: {harder_cell[:40]}... is harder to predict")
print(f"   Possible reason: Limited training data or more complex biology")

# ==================================================================
# ERROR PATTERN 2: By Phenotype
# ==================================================================
print("\n" + "="*70)
print("ERROR PATTERN #2: Performance by Phenotype")
print("="*70)

pheno_perf = pred_df.groupby('phenotype').agg({
    'correct_rf': 'mean',
    'correct_ensemble': 'mean',
    'gene': 'count'
}).rename(columns={'gene': 'count'})

print("\n" + pheno_perf.to_string())

# ==================================================================
# ERROR PATTERN 3: Misclassified Genes
# ==================================================================
print("\n" + "="*70)
print("ERROR PATTERN #3: Misclassified Genes")
print("="*70)

errors = pred_df[pred_df['correct_ensemble'] == 0]
print(f"\nTotal errors: {len(errors)} / {len(pred_df)} ({len(errors)/len(pred_df)*100:.1f}%)")

# Show some examples
print("\nExample misclassifications:")
print(errors[['gene', 'cell', 'true_label', 'pred_ensemble']].head(10).to_string(index=False))

# ==================================================================
# VISUALIZATIONS
# ==================================================================
print("\nGenerating visualizations...")

# Plot 1: Accuracy by cell line
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cell_names = [c[:30] + '...' if len(c) > 30 else c for c in cell_perf.index]
x = np.arange(len(cell_names))
width = 0.35

axes[0].bar(x - width/2, cell_perf['correct_rf']*100, width, label='RF Raw', color='steelblue')
axes[0].bar(x + width/2, cell_perf['correct_ensemble']*100, width, label='Ensemble', color='coral')
axes[0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0].set_title('Performance by Cell Line', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(cell_names, rotation=15, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([80, 100])

# Plot 2: Accuracy by phenotype
pheno_names = [p[:30] + '...' for p in pheno_perf.index]
x2 = np.arange(len(pheno_names))

axes[1].bar(x2 - width/2, pheno_perf['correct_rf']*100, width, label='RF Raw', color='steelblue')
axes[1].bar(x2 + width/2, pheno_perf['correct_ensemble']*100, width, label='Ensemble', color='coral')
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Performance by Phenotype', fontsize=12, fontweight='bold')
axes[1].set_xticks(x2)
axes[1].set_xticklabels(pheno_names, rotation=15, ha='right', fontsize=8)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_ylim([80, 100])

plt.tight_layout()
plt.savefig('../plots/error_patterns.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Plot saved: error_patterns.png")

# ==================================================================
# SUMMARY
# ==================================================================
print("\n" + "="*70)
print("ERROR ANALYSIS SUMMARY")
print("="*70)

print("\n📋 Key Findings:")
print(f"  1. Overall error rate: {(1-ens_acc)*100:.1f}%")
print(f"  2. Hardest cell line: {harder_cell[:40]}...")
print(f"  3. Ensemble reduces errors by {(ens_acc-rf_acc)*len(pred_df):.0f} examples")

print("\n⚠️  Implications:")
print("  • Some cell types are intrinsically harder to predict")
print("  • Need more diverse training data")
print("  • Ensemble helps but doesn't eliminate errors")

print("\n" + "="*70)
print("✅ ERROR ANALYSIS COMPLETE")
print("="*70)
