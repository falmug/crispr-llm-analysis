"""
IMPLEMENTATION: Context-Aware Embeddings

Current: Separate embeddings for gene, cell, phenotype, method
Proposed: Single contextualized embedding of entire scenario

Example:
  Instead of: embed("Bap1") + embed("glioblastoma") + embed("sensitivity")
  Use: embed("Knockout of Bap1 gene in glioblastoma cells increases sensitivity to gliocidin")
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from scipy import stats

print("="*70)
print("IMPLEMENTATION: CONTEXT-AWARE EMBEDDINGS")
print("="*70)

# Load benchmark
df = pd.read_csv("../data/benchmark_raw.csv")

print("\n[1/5] Loading embeddings...")

# Load SUMMARIZED embeddings (these are context-aware!)
embeddings = {}

embeddings['gliobl_pos'] = {
    'genes': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY/summarized_genes_mouse.npy", allow_pickle=True).item(),
    'cells': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY/summarized_cells.npy", allow_pickle=True).item(),
    'phenotypes': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY/summarized_phenotypes.npy", allow_pickle=True).item(),
    'methods': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY/summarized_methods.npy", allow_pickle=True).item(),
}

embeddings['gliobl_neg'] = {
    'genes': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY_FOR_INVERSE/summarized_genes_mouse.npy", allow_pickle=True).item(),
    'cells': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY_FOR_INVERSE/summarized_cells.npy", allow_pickle=True).item(),
    'phenotypes': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY_FOR_INVERSE/summarized_phenotypes.npy", allow_pickle=True).item(),
    'methods': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY_FOR_INVERSE/summarized_methods.npy", allow_pickle=True).item(),
}

embeddings['lung_pos'] = {
    'genes': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY/summarized_genes_mouse.npy", allow_pickle=True).item(),
    'cells': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY/summarized_cells.npy", allow_pickle=True).item(),
    'phenotypes': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY/summarized_phenotypes.npy", allow_pickle=True).item(),
    'methods': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY/summarized_methods.npy", allow_pickle=True).item(),
}

embeddings['lung_neg'] = {
    'genes': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY_FOR_INVERSE/summarized_genes_mouse.npy", allow_pickle=True).item(),
    'cells': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY_FOR_INVERSE/summarized_cells.npy", allow_pickle=True).item(),
    'phenotypes': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY_FOR_INVERSE/summarized_phenotypes.npy", allow_pickle=True).item(),
    'methods': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY_FOR_INVERSE/summarized_methods.npy", allow_pickle=True).item(),
}

# Also load standard embeddings for comparison
embeddings_standard = {}
embeddings_standard['gliobl_pos'] = {
    'genes': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY/genes_mouse.npy", allow_pickle=True).item(),
    'cells': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY/cells.npy", allow_pickle=True).item(),
    'phenotypes': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY/phenotypes.npy", allow_pickle=True).item(),
    'methods': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY/methods.npy", allow_pickle=True).item(),
}
embeddings_standard['gliobl_neg'] = {
    'genes': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY_FOR_INVERSE/genes_mouse.npy", allow_pickle=True).item(),
    'cells': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY_FOR_INVERSE/cells.npy", allow_pickle=True).item(),
    'phenotypes': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY_FOR_INVERSE/phenotypes.npy", allow_pickle=True).item(),
    'methods': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY_FOR_INVERSE/methods.npy", allow_pickle=True).item(),
}
embeddings_standard['lung_pos'] = {
    'genes': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY/genes_mouse.npy", allow_pickle=True).item(),
    'cells': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY/cells.npy", allow_pickle=True).item(),
    'phenotypes': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY/phenotypes.npy", allow_pickle=True).item(),
    'methods': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY/methods.npy", allow_pickle=True).item(),
}
embeddings_standard['lung_neg'] = {
    'genes': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY_FOR_INVERSE/genes_mouse.npy", allow_pickle=True).item(),
    'cells': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY_FOR_INVERSE/cells.npy", allow_pickle=True).item(),
    'phenotypes': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY_FOR_INVERSE/phenotypes.npy", allow_pickle=True).item(),
    'methods': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY_FOR_INVERSE/methods.npy", allow_pickle=True).item(),
}

print("  ✅ Loaded both standard and context-aware (summarized) embeddings")

def get_embedding_set(cell, phenotype, emb_dict):
    if 'glioblastoma' in cell.lower():
        if 'increased sensitivity' in phenotype:
            return emb_dict['gliobl_pos']
        else:
            return emb_dict['gliobl_neg']
    else:
        if 'increased resistance' in phenotype:
            return emb_dict['lung_pos']
        else:
            return emb_dict['lung_neg']

print("\n[2/5] Preparing features (both approaches)...")

# Standard embeddings
X_standard, y_standard = [], []
for idx, row in df.iterrows():
    gene = row['gene']
    cell = row['cell']
    method = row['perturbation'].capitalize()
    phenotype = row['phenotype']
    emb_set = get_embedding_set(cell, phenotype, embeddings_standard)
    if (gene in emb_set['genes'] and cell in emb_set['cells'] and 
        phenotype in emb_set['phenotypes'] and method in emb_set['methods']):
        X_standard.append(np.concatenate([
            emb_set['genes'][gene], emb_set['methods'][method],
            emb_set['cells'][cell], emb_set['phenotypes'][phenotype]
        ]))
        y_standard.append(row['hit'])

# Context-aware embeddings
X_context, y_context = [], []
for idx, row in df.iterrows():
    gene = row['gene']
    cell = row['cell']
    method = row['perturbation'].capitalize()
    phenotype = row['phenotype']
    emb_set = get_embedding_set(cell, phenotype, embeddings)
    if (gene in emb_set['genes'] and cell in emb_set['cells'] and 
        phenotype in emb_set['phenotypes'] and method in emb_set['methods']):
        X_context.append(np.concatenate([
            emb_set['genes'][gene], emb_set['methods'][method],
            emb_set['cells'][cell], emb_set['phenotypes'][phenotype]
        ]))
        y_context.append(row['hit'])

X_standard = np.array(X_standard)
y_standard = np.array(y_standard)
X_context = np.array(X_context)
y_context = np.array(y_context)

print(f"  Standard: {len(y_standard)} examples, dim={X_standard.shape[1]}")
print(f"  Context-aware: {len(y_context)} examples, dim={X_context.shape[1]}")

# Split both
X_train_std, X_temp_std, y_train_std, y_temp_std = train_test_split(
    X_standard, y_standard, test_size=0.4, random_state=42, stratify=y_standard)
X_val_std, X_test_std, y_val_std, y_test_std = train_test_split(
    X_temp_std, y_temp_std, test_size=0.5, random_state=42, stratify=y_temp_std)

X_train_ctx, X_temp_ctx, y_train_ctx, y_temp_ctx = train_test_split(
    X_context, y_context, test_size=0.4, random_state=42, stratify=y_context)
X_val_ctx, X_test_ctx, y_val_ctx, y_test_ctx = train_test_split(
    X_temp_ctx, y_temp_ctx, test_size=0.5, random_state=42, stratify=y_temp_ctx)

print("\n[3/5] Training models...")

# Standard approach
rf_std = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_std.fit(X_train_std, y_train_std)

# Context-aware approach
rf_ctx = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_ctx.fit(X_train_ctx, y_train_ctx)

print("  ✅ Both models trained")

print("\n[4/5] Evaluating...")

# Find thresholds
def find_threshold(model, X, y):
    pred = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, pred)
    return thresholds[np.argmax(tpr - fpr)]

threshold_std = find_threshold(rf_std, X_val_std, y_val_std)
threshold_ctx = find_threshold(rf_ctx, X_val_ctx, y_val_ctx)

# Test predictions
pred_std_proba = rf_std.predict_proba(X_test_std)[:, 1]
pred_ctx_proba = rf_ctx.predict_proba(X_test_ctx)[:, 1]

pred_std = (pred_std_proba >= threshold_std).astype(int)
pred_ctx = (pred_ctx_proba >= threshold_ctx).astype(int)

# Metrics
results = []
results.append({
    'Approach': 'Standard (Separate)',
    'F1': f1_score(y_test_std, pred_std),
    'AUROC': roc_auc_score(y_test_std, pred_std_proba),
    'Precision': precision_score(y_test_std, pred_std),
    'Recall': recall_score(y_test_std, pred_std),
    'FPR': ((pred_std == 1) & (y_test_std == 0)).sum() / (y_test_std == 0).sum()
})

results.append({
    'Approach': 'Context-Aware (Summarized)',
    'F1': f1_score(y_test_ctx, pred_ctx),
    'AUROC': roc_auc_score(y_test_ctx, pred_ctx_proba),
    'Precision': precision_score(y_test_ctx, pred_ctx),
    'Recall': recall_score(y_test_ctx, pred_ctx),
    'FPR': ((pred_ctx == 1) & (y_test_ctx == 0)).sum() / (y_test_ctx == 0).sum()
})

results_df = pd.DataFrame(results)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(results_df.to_string(index=False))

print("\n[5/5] Statistical significance test...")

def bootstrap_f1(y_true, y_pred_proba, threshold, n_iterations=1000):
    f1_scores = []
    n = len(y_true)
    np.random.seed(42)
    for _ in range(n_iterations):
        indices = np.random.choice(n, n, replace=True)
        y_boot = y_true[indices]
        pred_boot = y_pred_proba[indices]
        y_pred = (pred_boot >= threshold).astype(int)
        f1_scores.append(f1_score(y_boot, y_pred))
    return np.array(f1_scores)

std_boot = bootstrap_f1(y_test_std, pred_std_proba, threshold_std)
ctx_boot = bootstrap_f1(y_test_ctx, pred_ctx_proba, threshold_ctx)

print(f"\nStandard:      {std_boot.mean():.4f} [95% CI: {np.percentile(std_boot, 2.5):.4f}-{np.percentile(std_boot, 97.5):.4f}]")
print(f"Context-aware: {ctx_boot.mean():.4f} [95% CI: {np.percentile(ctx_boot, 2.5):.4f}-{np.percentile(ctx_boot, 97.5):.4f}]")

improvement = ctx_boot - std_boot
t_stat, p_value = stats.ttest_rel(ctx_boot, std_boot)

print(f"\nImprovement: {improvement.mean():.4f} ± {improvement.std():.4f}")
print(f"P-value: {p_value:.4f}")

print("\n" + "="*70)
if p_value < 0.05 and improvement.mean() > 0:
    print("✅ Context-aware embeddings ARE significantly better!")
    print(f"   Improvement: +{improvement.mean()*100:.2f}%")
elif p_value < 0.05 and improvement.mean() < 0:
    print("⚠️  Context-aware embeddings are significantly WORSE")
else:
    print("⚠️  No significant difference")
    print("   Possible reasons:")
    print("   - Summarized embeddings may not capture full context")
    print("   - Test set too small to detect small improvements")
    print("   - Context already implicit in concatenation")
print("="*70)

# Save results
results_df.to_csv('../results/context_aware_results.csv', index=False)
print("\n✅ Results saved to context_aware_results.csv")
