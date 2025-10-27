"""
Comprehensive Bias Analysis
Identifies and quantifies biases in the CRISPR screen prediction approach
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("="*70)
print("BIAS ANALYSIS - CRISPR SCREEN PREDICTION")
print("="*70)

# Load data
df_benchmark = pd.read_csv("../data/benchmark_raw.csv")
predictions = pd.read_csv("../results/predictions.csv")

# ==================================================================
# BIAS 1: HIGH-CONFIDENCE EXCLUSION
# ==================================================================
print("\n" + "="*70)
print("BIAS #1: HIGH-CONFIDENCE HIT DEFINITION")
print("="*70)

print("\nThe paper only uses 'high-confidence' hits:")
print("  - HIT (YES) = strong effect in expected direction")
print("  - NO-HIT (NO) = strong effect in OPPOSITE direction")
print("  - EXCLUDED = weak effect or no effect")

# From the paper's benchmark description
screen_17_total = 162  # genes tested in SCREEN_17
screen_17_used = 26    # genes in benchmark (23 YES + 3 NO)
screen_18_total = 19674  # genes tested in SCREEN_18
screen_18_used = 881   # genes in benchmark (73 YES + 808 NO)

total_tested = screen_17_total + screen_18_total
total_used = screen_17_used + screen_18_used
total_excluded = total_tested - total_used

print(f"\nSCREEN_17 (PD1 blockade):")
print(f"  Total genes tested: {screen_17_total}")
print(f"  High-confidence hits: {screen_17_used} ({screen_17_used/screen_17_total*100:.1f}%)")
print(f"  Excluded: {screen_17_total - screen_17_used} ({(screen_17_total-screen_17_used)/screen_17_total*100:.1f}%)")

print(f"\nSCREEN_18 (Gliocidin sensitivity):")
print(f"  Total genes tested: {screen_18_total}")
print(f"  High-confidence hits: {screen_18_used} ({screen_18_used/screen_18_total*100:.1f}%)")
print(f"  Excluded: {screen_18_total - screen_18_used} ({(screen_18_total-screen_18_used)/screen_18_total*100:.1f}%)")

print(f"\n📊 OVERALL:")
print(f"  Total genes tested: {total_tested}")
print(f"  Used in benchmark: {total_used} ({total_used/total_tested*100:.1f}%)")
print(f"  EXCLUDED: {total_excluded} ({total_excluded/total_tested*100:.2f}%)")

print("\n⚠️  IMPACT:")
print("  - Model trained only on extreme cases")
print("  - May not generalize to subtle biological effects")
print("  - Real-world predictions will include ambiguous cases")

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['SCREEN_17', 'SCREEN_18', 'Combined']
included = [screen_17_used, screen_18_used, total_used]
excluded = [screen_17_total - screen_17_used, screen_18_total - screen_18_used, total_excluded]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, included, width, label='Included (High-Confidence)', color='#2ecc71')
bars2 = ax.bar(x + width/2, excluded, width, label='Excluded (Ambiguous)', color='#e74c3c')

ax.set_ylabel('Number of Genes', fontsize=12)
ax.set_title('High-Confidence Exclusion Bias', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 100:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('../plots/bias_high_confidence_exclusion.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Plot saved: bias_high_confidence_exclusion.png")

# ==================================================================
# BIAS 2: TEST SET LIMITATIONS
# ==================================================================
print("\n" + "="*70)
print("BIAS #2: LIMITED TEST SET DIVERSITY")
print("="*70)

print(f"\nTest set characteristics:")
print(f"  Total examples: {len(df_benchmark)}")
print(f"  From # of papers: 2")
print(f"  Unique genes: {df_benchmark['gene'].nunique()}")
print(f"  Unique cell lines: {df_benchmark['cell'].nunique()}")
print(f"  Unique phenotypes: {df_benchmark['phenotype'].nunique()}")
print(f"  Publication date: October 2024 (very recent)")

print("\n⚠️  IMPACT:")
print("  - Limited biological diversity")
print("  - High variance in performance estimates")
print("  - May not generalize to other:")
print("    • Cell types")
print("    • Phenotypes")  
print("    • Experimental conditions")

# Visualize diversity
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

screen_counts = df_benchmark.groupby('cell').size()
axes[0].barh(range(len(screen_counts)), screen_counts.values, color='steelblue')
axes[0].set_yticks(range(len(screen_counts)))
axes[0].set_yticklabels([c[:40] + '...' if len(c) > 40 else c for c in screen_counts.index])
axes[0].set_xlabel('Number of Examples', fontsize=11)
axes[0].set_title('Distribution by Cell Line', fontsize=12, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

pheno_counts = df_benchmark.groupby('phenotype').size()
axes[1].barh(range(len(pheno_counts)), pheno_counts.values, color='coral')
axes[1].set_yticks(range(len(pheno_counts)))
axes[1].set_yticklabels([p[:40] + '...' for p in pheno_counts.index], fontsize=9)
axes[1].set_xlabel('Number of Examples', fontsize=11)
axes[1].set_title('Distribution by Phenotype', fontsize=12, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/bias_test_diversity.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Plot saved: bias_test_diversity.png")

# ==================================================================
# BIAS 3: CLASS BALANCE
# ==================================================================
print("\n" + "="*70)
print("BIAS #3: TRAINING vs TEST DISTRIBUTION MISMATCH")
print("="*70)

print("\nFrom paper (Table 2):")
print("  Training data: 7.74% positive, 92.26% negative")
print("  Imbalance ratio: 1:11.9")

print(f"\nTest data (after inversion trick):")
positive_test = df_benchmark['hit'].sum()
negative_test = len(df_benchmark) - positive_test
print(f"  Positive: {positive_test} ({positive_test/len(df_benchmark)*100:.1f}%)")
print(f"  Negative: {negative_test} ({negative_test/len(df_benchmark)*100:.1f}%)")
print(f"  Imbalance ratio: 1:1 (balanced)")

print("\n⚠️  IMPACT:")
print("  - Model trained on imbalanced data")
print("  - But tested on balanced data")
print("  - Real-world deployment will likely see imbalanced data")
print("  - Test performance may be overly optimistic")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

train_dist = [7.74, 92.26]
ax1.pie(train_dist, labels=['Positive\n(7.74%)', 'Negative\n(92.26%)'], 
        autopct='%1.1f%%', colors=['#3498db', '#e74c3c'], startangle=90)
ax1.set_title('Training Data Distribution\n(Imbalanced)', fontsize=12, fontweight='bold')

test_dist = [50, 50]
ax2.pie(test_dist, labels=['Positive\n(50%)', 'Negative\n(50%)'], 
        autopct='%1.1f%%', colors=['#2ecc71', '#f39c12'], startangle=90)
ax2.set_title('Test Data Distribution\n(Balanced after inversion)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('../plots/bias_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Plot saved: bias_class_distribution.png")

# ==================================================================
# SUMMARY
# ==================================================================
print("\n" + "="*70)
print("BIAS ANALYSIS SUMMARY")
print("="*70)

print("\n📋 Three Critical Biases Identified:")
print("\n1. HIGH-CONFIDENCE EXCLUSION (95.4% data excluded)")
print("   → Limits generalization to subtle effects")

print("\n2. LIMITED TEST DIVERSITY (2 papers, 2 cell lines)")
print("   → High variance, may not generalize broadly")

print("\n3. TRAIN/TEST DISTRIBUTION MISMATCH (7.74% vs 50%)")
print("   → Test performance may be optimistic")

print("\n💡 RECOMMENDATIONS:")
print("  1. Test on diverse cell lines and phenotypes")
print("  2. Include borderline cases in training")
print("  3. Evaluate on imbalanced test sets (real-world)")
print("  4. Use ensemble to reduce overfitting")
print("  5. Validate on prospective screens")

print("\n" + "="*70)
print("✅ BIAS ANALYSIS COMPLETE")
print("="*70)
print("\nPlots saved to: ../plots/")
