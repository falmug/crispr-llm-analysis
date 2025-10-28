"""
Analysis using official organizer-provided embeddings
Uses gene NAMES directly (not IDs)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from scipy import stats

print("="*70)
print("ANALYSIS WITH OFFICIAL EMBEDDINGS")
print("="*70)

# Load benchmark
print("\n[1/6] Loading benchmark...")
df = pd.read_csv("../data/benchmark_raw.csv")
print(f"  Total examples: {len(df)}")

# Load official embeddings
print("\n[2/6] Loading official organizer embeddings...")

# SCREEN_18: Glioblastoma (881 genes)
genes_18 = np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY/genes_mouse.npy", allow_pickle=True).item()
cells_18 = np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY/cells.npy", allow_pickle=True).item()
pheno_18 = np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY/phenotypes.npy", allow_pickle=True).item()
methods_18 = np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY/methods.npy", allow_pickle=True).item()

# SCREEN_17: Lung carcinoma (26 genes)
genes_17 = np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY/genes_mouse.npy", allow_pickle=True).item()
cells_17 = np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY/cells.npy", allow_pickle=True).item()
pheno_17 = np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY/phenotypes.npy", allow_pickle=True).item()
methods_17 = np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY/methods.npy", allow_pickle=True).item()

print(f"  ✅ SCREEN_18 (Glioblastoma): {len(genes_18)} genes")
print(f"  ✅ SCREEN_17 (Lung carcinoma): {len(genes_17)} genes")

# Prepare features
print("\n[3/6] Preparing features...")
X, y, metadata = [], [], []

for idx, row in df.iterrows():
    gene = row['gene']
    cell = row['cell']
    method = row['perturbation']
    phenotype = row['phenotype']
    
    # Select appropriate embedding set based on cell line
    if 'glioblastoma' in cell.lower():
        gene_dict = genes_18
        cell_dict = cells_18
        pheno_dict = pheno_18
        method_dict = methods_18
    else:  # lung carcinoma
        gene_dict = genes_17
        cell_dict = cells_17
        pheno_dict = pheno_17
        method_dict = methods_17
    
    # Check if all embeddings exist
    if (gene in gene_dict and cell in cell_dict and 
        phenotype in pheno_dict and method in method_dict):
        
        # Concatenate all 4 embeddings
        X.append(np.concatenate([
            gene_dict[gene],
            method_dict[method],
            cell_dict[cell],
            pheno_dict[phenotype]
        ]))
        
        y.append(row['hit'])
        metadata.append({'gene': gene, 'cell': cell, 'phenotype': phenotype})

X = np.array(X)
y = np.array(y)

print(f"  ✅ Valid examples: {len(y)}/{len(df)} ({len(y)/len(df)*100:.1f}%)")
print(f"  ✅ Feature dimension: {X.shape[1]}")

# Split: 60% train, 20% val, 20% test
print("\n[4/6] Creating train/val/test split...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"  Train: {len(y_train)} ({len(y_train)/len(y)*100:.0f}%)")
print(f"  Val:   {len(y_val)} ({len(y_val)/len(y)*100:.0f}%)")
print(f"  Test:  {len(y_test)} ({len(y_test)/len(y)*100:.0f}%)")

# Train models
print("\n[5/6] Training models...")

rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print("  ✅ Random Forest")

gb = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
gb.fit(X_train, y_train)
print("  ✅ Gradient Boosting")

mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
print("  ✅ MLP Neural Network")

# Find optimal thresholds on validation set
def find_optimal_threshold(model, X_val, y_val):
    pred_proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]

threshold_rf = find_optimal_threshold(rf, X_val, y_val)
threshold_gb = find_optimal_threshold(gb, X_val, y_val)
threshold_mlp = find_optimal_threshold(mlp, X_val, y_val)

print(f"\n  Validation-tuned thresholds:")
print(f"    RF: {threshold_rf:.3f}")
print(f"    GB: {threshold_gb:.3f}")
print(f"    MLP: {threshold_mlp:.3f}")

# Test set evaluation
print("\n[6/6] Evaluating on test set...")

def evaluate(model, X, y, threshold, name):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = (pred_proba >= threshold).astype(int)
    
    return {
        'Model': name,
        'AUROC': roc_auc_score(y, pred_proba),
        'F1': f1_score(y, pred),
        'Precision': precision_score(y, pred),
        'Recall': recall_score(y, pred),
        'Threshold': threshold
    }

results = []
results.append(evaluate(rf, X_test, y_test, threshold_rf, "RF"))
results.append(evaluate(gb, X_test, y_test, threshold_gb, "GB"))
results.append(evaluate(mlp, X_test, y_test, threshold_mlp, "MLP"))

# Ensemble
pred_rf = rf.predict_proba(X_test)[:, 1]
pred_gb = gb.predict_proba(X_test)[:, 1]
pred_mlp = mlp.predict_proba(X_test)[:, 1]
pred_ensemble = (pred_rf + pred_gb + pred_mlp) / 3

# Find ensemble threshold on validation
val_pred_rf = rf.predict_proba(X_val)[:, 1]
val_pred_gb = gb.predict_proba(X_val)[:, 1]
val_pred_mlp = mlp.predict_proba(X_val)[:, 1]
val_pred_ensemble = (val_pred_rf + val_pred_gb + val_pred_mlp) / 3

fpr, tpr, thresholds = roc_curve(y_val, val_pred_ensemble)
threshold_ens = thresholds[np.argmax(tpr - fpr)]

results.append({
    'Model': 'Ensemble (3 models)',
    'AUROC': roc_auc_score(y_test, pred_ensemble),
    'F1': f1_score(y_test, (pred_ensemble >= threshold_ens).astype(int)),
    'Precision': precision_score(y_test, (pred_ensemble >= threshold_ens).astype(int)),
    'Recall': recall_score(y_test, (pred_ensemble >= threshold_ens).astype(int)),
    'Threshold': threshold_ens
})

results_df = pd.DataFrame(results)

print("\n" + "="*70)
print("RESULTS (Official Embeddings)")
print("="*70)
print(results_df.to_string(index=False))

# Statistical validation
print("\n" + "="*70)
print("STATISTICAL VALIDATION")
print("="*70)

def bootstrap_f1(y_true, y_pred_proba, threshold, n_iterations=1000):
    f1_scores = []
    n = len(y_true)
    for _ in range(n_iterations):
        indices = np.random.choice(n, n, replace=True)
        y_boot = y_true[indices]
        pred_boot = y_pred_proba[indices]
        y_pred = (pred_boot >= threshold).astype(int)
        f1_scores.append(f1_score(y_boot, y_pred))
    return np.array(f1_scores)

print("\nBootstrapping (1000 iterations)...")
rf_boot = bootstrap_f1(y_test, pred_rf, threshold_rf)
ens_boot = bootstrap_f1(y_test, pred_ensemble, threshold_ens)

print(f"RF:       {rf_boot.mean():.4f} [95% CI: {np.percentile(rf_boot, 2.5):.4f}-{np.percentile(rf_boot, 97.5):.4f}]")
print(f"Ensemble: {ens_boot.mean():.4f} [95% CI: {np.percentile(ens_boot, 2.5):.4f}-{np.percentile(ens_boot, 97.5):.4f}]")

improvement = ens_boot - rf_boot
t_stat, p_value = stats.ttest_rel(ens_boot, rf_boot)

print(f"\nImprovement: {improvement.mean():.4f} ± {improvement.std():.4f}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.05:
    print("✅ Ensemble improvement IS statistically significant!")
else:
    print("⚠️  Ensemble improvement NOT statistically significant")

# Save
results_df.to_csv('../results/official_embeddings_results.csv', index=False)

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE WITH OFFICIAL EMBEDDINGS")
print("="*70)
