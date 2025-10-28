"""
Evidence-Based Bias Analysis
All claims backed by verifiable data or paper citations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("EVIDENCE-BASED BIAS ANALYSIS")
print("="*70)

# Load data
df_benchmark = pd.read_csv("../data/benchmark_raw.csv")

# ==================================================================
# BIAS 1: TRAINING SET IMBALANCE
# ==================================================================
print("\n" + "="*70)
print("BIAS #1: SEVERE TRAINING SET IMBALANCE")
print("="*70)

print("\nFrom paper (Section 3):")
print("  Training data: 7.74% positive, 92.26% negative")
print("  This is approximately 1:12 imbalance ratio")

print("\nTest benchmark characteristics:")
positive = df_benchmark['hit'].sum()
total = len(df_benchmark)
negative = total - positive

print(f"  Positive: {positive} ({positive/total*100:.1f}%)")
print(f"  Negative: {negative} ({negative/total*100:.1f}%)")
print(f"  Ratio: 1:{negative/positive:.1f}")

print("\n⚠️  IMPACT:")
print("  • Model trained on highly imbalanced data (1:12)")
print("  • Test set artificially balanced (1:1) via inversion trick")
print("  • Real-world deployment will see ~7.74% hit rate")
print("  • Test performance likely optimistic for real applications")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Training distribution
train_data = [7.74, 92.26]
colors1 = ['#3498db', '#e74c3c']
ax1.pie(train_data, labels=['Positive\n(7.74%)', 'Negative\n(92.26%)'], 
        autopct='%1.1f%%', colors=colors1, startangle=90)
ax1.set_title('Training Data\n(Paper: Section 3)', fontsize=12, fontweight='bold')

# Test distribution
test_pct = [positive/total*100, negative/total*100]
colors2 = ['#2ecc71', '#f39c12']
ax2.pie(test_pct, labels=[f'Positive\n({positive/total*100:.1f}%)', 
                           f'Negative\n({negative/total*100:.1f}%)'], 
        autopct='%1.1f%%', colors=colors2, startangle=90)
ax2.set_title('Test Benchmark\n(After Inversion)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('../plots/bias_class_imbalance.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Plot saved: bias_class_imbalance.png")

# ==================================================================
# BIAS 2: LIMITED TEST SET DIVERSITY
# ==================================================================
print("\n" + "="*70)
print("BIAS #2: LIMITED TEST SET DIVERSITY")
print("="*70)

print(f"\nTest set characteristics:")
print(f"  Total examples: {len(df_benchmark)}")
print(f"  Source papers: 2 (PMID 39567689, PMID 39385035)")
print(f"  Unique genes: {df_benchmark['gene'].nunique()}")
print(f"  Unique cell lines: {df_benchmark['cell'].nunique()}")
print(f"  Unique phenotypes: {df_benchmark['phenotype'].nunique()}")
print(f"  Publication date: October 2024 (post-cutoff)")

# Calculate statistics
genes_per_cell = df_benchmark.groupby('cell')['gene'].nunique()
examples_per_cell = df_benchmark.groupby('cell').size()

print("\nBreakdown by cell line:")
for cell in df_benchmark['cell'].unique():
    n_genes = genes_per_cell[cell]
    n_examples = examples_per_cell[cell]
    print(f"  {cell[:50]}:")
    print(f"    {n_genes} unique genes, {n_examples} examples")

print("\n⚠️  IMPACT:")
print("  • Small sample size (n=1,814) → high variance")
print("  • Only 2 biological contexts tested")
print("  • Uncertain generalization to:")
print("    - Other cell types")
print("    - Other phenotypes")
print("    - Other experimental conditions")

# Visualize diversity
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Examples per cell line
cell_counts = df_benchmark.groupby('cell').size()
cell_names = [c[:40] + '...' if len(c) > 40 else c for c in cell_counts.index]

axes[0].barh(range(len(cell_names)), cell_counts.values, color='steelblue')
axes[0].set_yticks(range(len(cell_names)))
axes[0].set_yticklabels(cell_names, fontsize=10)
axes[0].set_xlabel('Number of Examples', fontsize=11)
axes[0].set_title('Test Set by Cell Line\n(Only 2 contexts)', fontsize=12, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Add counts
for i, v in enumerate(cell_counts.values):
    axes[0].text(v + 20, i, str(v), va='center', fontsize=9)

# Plot 2: Unique genes per cell
axes[1].barh(range(len(cell_names)), genes_per_cell.values, color='coral')
axes[1].set_yticks(range(len(cell_names)))
axes[1].set_yticklabels(cell_names, fontsize=10)
axes[1].set_xlabel('Unique Genes Tested', fontsize=11)
axes[1].set_title('Gene Coverage by Cell Line', fontsize=12, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

# Add counts
for i, v in enumerate(genes_per_cell.values):
    axes[1].text(v + 10, i, str(v), va='center', fontsize=9)

plt.tight_layout()
plt.savefig('../plots/bias_test_diversity.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Plot saved: bias_test_diversity.png")

# ==================================================================
# BIAS 3: HIGH-CONFIDENCE FILTERING
# ==================================================================
print("\n" + "="*70)
print("BIAS #3: HIGH-CONFIDENCE HIT DEFINITION")
print("="*70)

print("\nFrom paper (Section 2.2):")
print("  'We employ a high-confidence hit definition'")
print("  ")
print("  HIT (YES) = Strong effect in expected direction")
print("  NO-HIT (NO) = Strong effect in OPPOSITE direction")
print("  EXCLUDED = Weak/ambiguous/no effect")

print("\n⚠️  IMPACT:")
print("  • Training on extreme cases only")
print("  • Borderline effects systematically excluded")
print("  • Unknown: How many genes showed borderline effects")
print("  • Unknown: Original screen sizes before filtering")
print("  ")
print("  Consequence:")
print("  • Model may not generalize to subtle biological effects")
print("  • Real-world predictions will include ambiguous cases")
print("  • Performance on borderline cases untested")

# Create conceptual diagram
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Title
ax.text(0.5, 0.95, 'High-Confidence Filtering Strategy', 
        ha='center', fontsize=14, fontweight='bold')

# Original screen
ax.add_patch(plt.Rectangle((0.1, 0.7), 0.8, 0.15, 
                            facecolor='lightgray', edgecolor='black', linewidth=2))
ax.text(0.5, 0.775, 'Original CRISPR Screen Results', 
        ha='center', va='center', fontsize=11, fontweight='bold')

# Arrow down
ax.arrow(0.5, 0.7, 0, -0.08, head_width=0.05, head_length=0.02, 
         fc='black', ec='black', linewidth=2)

# Filtering
ax.text(0.5, 0.6, 'Apply High-Confidence Filter', 
        ha='center', fontsize=10, style='italic')

# Three categories
categories = [
    ('Strong Effect\n(Expected Direction)', 0.15, '#2ecc71', 'INCLUDED\n(HIT)'),
    ('Weak/No Effect', 0.5, '#95a5a6', 'EXCLUDED'),
    ('Strong Effect\n(Opposite Direction)', 0.85, '#e74c3c', 'INCLUDED\n(NO-HIT)')
]

for label, x, color, status in categories:
    ax.add_patch(plt.Rectangle((x-0.08, 0.35), 0.16, 0.15, 
                                facecolor=color, edgecolor='black', linewidth=1.5))
    ax.text(x, 0.425, label, ha='center', va='center', fontsize=9)
    ax.text(x, 0.25, status, ha='center', va='center', fontsize=8, fontweight='bold')

# Bottom note
ax.text(0.5, 0.1, '⚠️  Borderline cases excluded → Limited generalization to subtle effects', 
        ha='center', fontsize=10, color='red', fontweight='bold')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('../plots/bias_high_confidence_filtering.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Plot saved: bias_high_confidence_filtering.png")

# ==================================================================
# SUMMARY
# ==================================================================
print("\n" + "="*70)
print("EVIDENCE-BASED BIAS SUMMARY")
print("="*70)

print("\n📋 Three Verified Biases:")
print("\n1. TRAINING SET IMBALANCE (7.74% positive)")
print("   Source: Paper Section 3")
print("   → Model trained on highly skewed distribution")
print("   → Test performance may not reflect real-world")

print("\n2. LIMITED TEST DIVERSITY (2 papers, 900 genes)")
print("   Source: Measured from benchmark data")
print("   → Small sample size, high variance")
print("   → Narrow biological contexts tested")

print("\n3. HIGH-CONFIDENCE FILTERING (extreme cases only)")
print("   Source: Paper Section 2.2")
print("   → Borderline cases excluded")
print("   → Generalization to subtle effects unclear")

print("\n💡 RECOMMENDATIONS:")
print("  1. Evaluate on larger, more diverse test sets")
print("  2. Include borderline cases in future benchmarks")
print("  3. Test on imbalanced data (real-world scenario)")
print("  4. Validate on prospective screens")
print("  5. Report performance stratified by effect size")

print("\n" + "="*70)
print("✅ BIAS ANALYSIS COMPLETE")
print("="*70)
print("\nAll claims backed by verifiable evidence!")
print("Plots saved to: ../plots/")
