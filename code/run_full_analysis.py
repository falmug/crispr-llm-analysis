"""
Complete CRISPR Screen Analysis
Uses all 4 embeddings: gene + method + cell + phenotype
"""

import sys
sys.path.append("../../../benchmarks")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("CRISPR SCREEN PREDICTION - FULL ANALYSIS")
print("="*70)

# Load genome mapping
print("\n[1/7] Loading genome mapping...")
genome = pd.read_csv("../../../genomes/genome_mus_musculus.tsv", sep="\t")
gene_name_to_id = dict(zip(genome['OFFICIAL_SYMBOL'], genome['IDENTIFIER_ID']))

# Load benchmark
print("\n[2/7] Loading benchmark data...")
df = pd.read_csv("../data/benchmark_raw.csv")
print(f"  Total: {len(df)} examples")

# Load ALL embeddings
print("\n[3/7] Loading embeddings...")
gene_emb_raw = np.load("../../data/embeddings/genes_mouse.npy", allow_pickle=True).item()
gene_emb_summ = np.load("../../data/embeddings/summarized_genes_mouse.npy", allow_pickle=True).item()
method_emb_raw = np.load("../../data/embeddings/methods.npy", allow_pickle=True).item()
method_emb_summ = np.load("../../data/embeddings/summarized_methods.npy", allow_pickle=True).item()
cell_emb = np.load("../../data/embeddings/benchmark_cells.npy", allow_pickle=True).item()
pheno_emb = np.load("../../data/embeddings/benchmark_phenotypes.npy", allow_pickle=True).item()

print(f"  ✅ Gene embeddings: {len(gene_emb_raw)}")
print(f"  ✅ Cell embeddings: {len(cell_emb)}")
print(f"  ✅ Phenotype embeddings: {len(pheno_emb)}")

# Prepare features with ALL 4 embeddings
print("\n[4/7] Preparing features (gene + method + cell + phenotype)...")
X_raw, X_summ, y, metadata = [], [], [], []

for idx, row in df.iterrows():
    gene_name = row['gene']
    method = row['perturbation'].title()
    cell = row['cell']
    phenotype = row['phenotype']
    
    # Map gene name to ID
    if gene_name not in gene_name_to_id:
        continue
    gene_id = gene_name_to_id[gene_name]
    
    # Check all embeddings exist
    if (gene_id in gene_emb_raw and method in method_emb_raw and 
        cell in cell_emb and phenotype in pheno_emb):
        
        # Concatenate all 4 embeddings (12,288 dims total)
        X_raw.append(np.concatenate([
            gene_emb_raw[gene_id],    # 3072
            method_emb_raw[method],    # 3072
            cell_emb[cell],            # 3072
            pheno_emb[phenotype]       # 3072
        ]))
        
        X_summ.append(np.concatenate([
            gene_emb_summ[gene_id],
            method_emb_summ[method],
            cell_emb[cell],
            pheno_emb[phenotype]
        ]))
        
        y.append(row['hit'])
        metadata.append({
            'gene': gene_name,
            'cell': cell,
            'phenotype': phenotype
        })

X_raw = np.array(X_raw)
X_summ = np.array(X_summ)
y = np.array(y)

print(f"  ✅ Valid: {len(y)}/{len(df)} ({len(y)/len(df)*100:.1f}%)")
print(f"  ✅ Feature dimension: {X_raw.shape[1]} (expected: 12,288)")

# Split data
print("\n[5/7] Training models...")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.3, random_state=42, stratify=y
)
X_train_summ, X_test_summ = train_test_split(
    X_summ, test_size=0.3, random_state=42, stratify=y
)[0:2]

meta_train, meta_test = train_test_split(
    metadata, test_size=0.3, random_state=42, stratify=y
)

print(f"  Train: {len(y_train)} | Test: {len(y_test)}")

# Train models
print("\n  Training Raw Embedding models:")
rf_raw = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_raw.fit(X_train_raw, y_train)
print("    ✅ Random Forest")

gb_raw = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
gb_raw.fit(X_train_raw, y_train)
print("    ✅ Gradient Boosting")

print("\n  Training Summary Embedding models:")
rf_summ = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_summ.fit(X_train_summ, y_train)
print("    ✅ Random Forest")

gb_summ = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
gb_summ.fit(X_train_summ, y_train)
print("    ✅ Gradient Boosting")

# Predictions
print("\n[6/7] Generating predictions & ensembles...")
pred_rf_raw = rf_raw.predict_proba(X_test_raw)[:, 1]
pred_gb_raw = gb_raw.predict_proba(X_test_raw)[:, 1]
pred_rf_summ = rf_summ.predict_proba(X_test_summ)[:, 1]
pred_gb_summ = gb_summ.predict_proba(X_test_summ)[:, 1]

# Ensemble strategies
pred_ensemble_simple = (pred_rf_raw + pred_rf_summ) / 2
pred_ensemble_all = (pred_rf_raw + pred_gb_raw + pred_rf_summ + pred_gb_summ) / 4
pred_ensemble_best = 0.35*pred_rf_raw + 0.35*pred_gb_raw + 0.15*pred_rf_summ + 0.15*pred_gb_summ

# Compute metrics
def compute_metrics(y_true, y_pred_proba, name="Model"):
    auroc = roc_auc_score(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    y_pred = (y_pred_proba >= thresholds[optimal_idx]).astype(int)
    
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'Model': name,
        'AUROC': auroc,
        'AUPRC': auprc,
        'F1': f1,
        'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0
    }

results = []
results.append(compute_metrics(y_test, pred_rf_raw, "RF Raw"))
results.append(compute_metrics(y_test, pred_gb_raw, "GB Raw"))
results.append(compute_metrics(y_test, pred_rf_summ, "RF Summary"))
results.append(compute_metrics(y_test, pred_gb_summ, "GB Summary"))
results.append(compute_metrics(y_test, pred_ensemble_simple, "Ensemble (Simple)"))
results.append(compute_metrics(y_test, pred_ensemble_all, "Ensemble (All)"))
results.append(compute_metrics(y_test, pred_ensemble_best, "Ensemble (Weighted)"))

results_df = pd.DataFrame(results)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(results_df[['Model', 'F1', 'AUROC', 'FPR', 'Precision', 'Recall']].to_string(index=False))

# Save everything
results_df.to_csv("../results/model_comparison.csv", index=False)

pred_df = pd.DataFrame(meta_test)
pred_df['true_label'] = y_test
pred_df['pred_rf_raw'] = pred_rf_raw
pred_df['pred_ensemble'] = pred_ensemble_best
pred_df.to_csv("../results/predictions.csv", index=False)

print("\n✅ Results saved to ../results/")

# Generate plots
print("\n[7/7] Generating visualizations...")

plt.figure(figsize=(10, 7))
for name, preds in [('RF Raw', pred_rf_raw), 
                    ('RF Summary', pred_rf_summ),
                    ('Ensemble', pred_ensemble_best)]:
    fpr_c, tpr_c, _ = roc_curve(y_test, preds)
    plt.plot(fpr_c, tpr_c, label=f'{name} (AUROC={roc_auc_score(y_test, preds):.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Full Model (All 4 Embeddings)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../plots/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
top5 = results_df.nlargest(5, 'F1')
plt.barh(top5['Model'], top5['F1'], color='steelblue')
plt.xlabel('F1 Score', fontsize=12)
plt.title('Top 5 Models - F1 Scores', fontsize=14, fontweight='bold')
plt.xlim([0, 1])
for i, (model, f1) in enumerate(zip(top5['Model'], top5['F1'])):
    plt.text(f1 + 0.01, i, f'{f1:.3f}', va='center', fontsize=10)
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('../plots/f1_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✅ Plots saved")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)

best_idx = results_df['F1'].idxmax()
best = results_df.iloc[best_idx]
baseline = results_df[results_df['Model'] == 'RF Raw'].iloc[0]

print(f"\n🏆 Best Model: {best['Model']}")
print(f"   F1: {best['F1']:.3f}")
print(f"   AUROC: {best['AUROC']:.3f}")
print(f"   FPR: {best['FPR']:.3f}")

improvement = (best['F1'] - baseline['F1']) / baseline['F1'] * 100
print(f"\n📈 Improvement over baseline: {improvement:+.1f}%")

print("\n✅ Ready for deep bias analysis!")
