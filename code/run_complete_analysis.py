"""
Complete analysis pipeline for CRISPR screen prediction
Uses gene name → ID mapping to access embeddings
"""

import sys
sys.path.append("../../")
sys.path.append("../../../benchmarks")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("CRISPR SCREEN PREDICTION - COMPLETE ANALYSIS")
print("="*70)

# Load genome for gene name → ID mapping
print("\n[1/8] Loading genome mapping...")
genome = pd.read_csv("../../../genomes/genome_mus_musculus.tsv", sep="\t")
gene_name_to_id = dict(zip(genome['OFFICIAL_SYMBOL'], genome['IDENTIFIER_ID']))
print(f"  Loaded {len(gene_name_to_id)} gene mappings")

# Load benchmark data
print("\n[2/8] Loading benchmark data...")
df = pd.read_csv("../data/benchmark_raw.csv")
print(f"  Loaded {len(df)} examples")
print(f"  Positive: {df['hit'].sum()} ({df['hit'].mean()*100:.1f}%)")
print(f"  Negative: {(df['hit']==0).sum()} ({(1-df['hit'].mean())*100:.1f}%)")

# Load embeddings
print("\n[3/8] Loading pre-computed embeddings...")
gene_emb_raw = np.load("../../data/embeddings/genes_mouse.npy", allow_pickle=True).item()
gene_emb_summ = np.load("../../data/embeddings/summarized_genes_mouse.npy", allow_pickle=True).item()
method_emb_raw = np.load("../../data/embeddings/methods.npy", allow_pickle=True).item()
method_emb_summ = np.load("../../data/embeddings/summarized_methods.npy", allow_pickle=True).item()

print(f"  Loaded {len(gene_emb_raw)} gene embeddings")
print(f"  Embedding dimension: {list(gene_emb_raw.values())[0].shape[0]}")

print("\n[4/8] Preparing features...")
X_raw = []
X_summ = []
y = []
valid_indices = []

for idx, row in df.iterrows():
    gene_name = row['gene']
    method = row['perturbation'].title()
    
    # Map gene name to ID
    if gene_name not in gene_name_to_id:
        continue
    gene_id = gene_name_to_id[gene_name]
    
    # Check if we have embeddings
    if gene_id in gene_emb_raw and method in method_emb_raw:
        # Raw embeddings
        gene_e = gene_emb_raw[gene_id]
        method_e = method_emb_raw[method]
        X_raw.append(np.concatenate([gene_e, method_e]))
        
        # Summary embeddings
        gene_e_s = gene_emb_summ[gene_id]
        method_e_s = method_emb_summ[method]
        X_summ.append(np.concatenate([gene_e_s, method_e_s]))
        
        y.append(row['hit'])
        valid_indices.append(idx)

X_raw = np.array(X_raw)
X_summ = np.array(X_summ)
y = np.array(y)

print(f"  Valid examples: {len(y)} / {len(df)} ({len(y)/len(df)*100:.1f}%)")
print(f"  Feature dimension: {X_raw.shape[1]}")

print("\n[5/8] Training models...")
# Split data
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.3, random_state=42, stratify=y
)
X_train_summ, X_test_summ, _, _ = train_test_split(
    X_summ, y, test_size=0.3, random_state=42, stratify=y
)

# Train raw model
print("  Training raw embedding model...")
model_raw = LogisticRegression(max_iter=1000, random_state=42)
model_raw.fit(X_train_raw, y_train)

# Train summary model
print("  Training summary embedding model...")
model_summ = LogisticRegression(max_iter=1000, random_state=42)
model_summ.fit(X_train_summ, y_train)

print("\n[6/8] Generating predictions...")
pred_raw = model_raw.predict_proba(X_test_raw)[:, 1]
pred_summ = model_summ.predict_proba(X_test_summ)[:, 1]

# Ensemble
pred_ensemble_avg = (pred_raw + pred_summ) / 2
pred_ensemble_weighted = 0.6 * pred_raw + 0.4 * pred_summ

print("\n[7/8] Computing metrics...")
def compute_metrics(y_true, y_pred_proba):
    auroc = roc_auc_score(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr_val = fp / (fp + tn)
    
    return {
        'AUROC': auroc,
        'AUPRC': auprc,
        'F1': f1,
        'FPR': fpr_val,
        'Threshold': optimal_threshold
    }

results = {}
results['Raw Embedding'] = compute_metrics(y_test, pred_raw)
results['Summary Embedding'] = compute_metrics(y_test, pred_summ)
results['Ensemble (Avg)'] = compute_metrics(y_test, pred_ensemble_avg)
results['Ensemble (Weighted)'] = compute_metrics(y_test, pred_ensemble_weighted)

# Print results
print("\n" + "="*70)
print("RESULTS")
print("="*70)
results_df = pd.DataFrame(results).T
print(results_df.to_string())

# Save results
results_df.to_csv("../results/model_comparison.csv")
print("\n✅ Results saved to ../results/model_comparison.csv")

print("\n[8/8] Generating plots...")
# ROC curves
plt.figure(figsize=(10, 6))
for name, preds in [('Raw', pred_raw), ('Summary', pred_summ), 
                    ('Ensemble', pred_ensemble_avg)]:
    fpr_curve, tpr_curve, _ = roc_curve(y_test, preds)
    auroc = roc_auc_score(y_test, preds)
    plt.plot(fpr_curve, tpr_curve, label=f'{name} (AUROC={auroc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('../plots/roc_curves.png', dpi=300, bbox_inches='tight')
print("  ✅ ROC curves saved")

# F1 comparison
plt.figure(figsize=(8, 6))
models = ['Raw', 'Summary', 'Ensemble\n(Avg)', 'Ensemble\n(Weighted)']
f1_scores = [results['Raw Embedding']['F1'], 
             results['Summary Embedding']['F1'],
             results['Ensemble (Avg)']['F1'],
             results['Ensemble (Weighted)']['F1']]

plt.bar(models, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.ylabel('F1 Score', fontsize=12)
plt.title('Model Comparison - F1 Scores', fontsize=14, fontweight='bold')
plt.ylim([0, 1])
for i, v in enumerate(f1_scores):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.savefig('../plots/f1_comparison.png', dpi=300, bbox_inches='tight')
print("  ✅ F1 comparison saved")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print("\nKey findings:")
best_model = results_df['F1'].idxmax()
improvement = (results_df.loc[best_model, 'F1'] - results_df.loc['Raw Embedding', 'F1']) / results_df.loc['Raw Embedding', 'F1'] * 100
print(f"✅ Best model: {best_model}")
print(f"✅ F1 Score: {results_df.loc[best_model, 'F1']:.3f}")
print(f"✅ Improvement over raw: {improvement:.1f}%")
print(f"✅ AUROC: {results_df.loc[best_model, 'AUROC']:.3f}")
print(f"✅ FPR: {results_df.loc[best_model, 'FPR']:.3f}")
