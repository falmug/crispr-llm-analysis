"""
Proper evaluation with train/val/test split
NO test set leakage - threshold chosen on validation set
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from scipy import stats
import matplotlib.pyplot as plt

print("="*70)
print("PROPER EVALUATION - NO TEST SET LEAKAGE")
print("="*70)

# Load genome mapping
print("\n[1/6] Loading data...")
genome = pd.read_csv("../data/genome_mus_musculus.tsv", sep="\t")
gene_name_to_id = dict(zip(genome['OFFICIAL_SYMBOL'], genome['IDENTIFIER_ID']))

# Load benchmark
df = pd.read_csv("../data/benchmark_raw.csv")

# Load embeddings
gene_emb_raw = np.load("../../data/embeddings/genes_mouse.npy", allow_pickle=True).item()
gene_emb_summ = np.load("../../data/embeddings/summarized_genes_mouse.npy", allow_pickle=True).item()
method_emb_raw = np.load("../../data/embeddings/methods.npy", allow_pickle=True).item()
method_emb_summ = np.load("../../data/embeddings/summarized_methods.npy", allow_pickle=True).item()
cell_emb = np.load("../../data/embeddings/benchmark_cells.npy", allow_pickle=True).item()
pheno_emb = np.load("../../data/embeddings/benchmark_phenotypes.npy", allow_pickle=True).item()

# Prepare features
print("\n[2/6] Preparing features...")
X_raw, X_summ, y, metadata = [], [], [], []

for idx, row in df.iterrows():
    gene_name = row['gene']
    method = row['perturbation'].title()
    cell = row['cell']
    phenotype = row['phenotype']
    
    if gene_name not in gene_name_to_id:
        continue
    gene_id = gene_name_to_id[gene_name]
    
    if (gene_id in gene_emb_raw and method in method_emb_raw and 
        cell in cell_emb and phenotype in pheno_emb):
        
        X_raw.append(np.concatenate([
            gene_emb_raw[gene_id],
            method_emb_raw[method],
            cell_emb[cell],
            pheno_emb[phenotype]
        ]))
        
        X_summ.append(np.concatenate([
            gene_emb_summ[gene_id],
            method_emb_summ[method],
            cell_emb[cell],
            pheno_emb[phenotype]
        ]))
        
        y.append(row['hit'])
        metadata.append({'gene': gene_name, 'cell': cell, 'phenotype': phenotype})

X_raw = np.array(X_raw)
X_summ = np.array(X_summ)
y = np.array(y)

print(f"  Total examples: {len(y)}")

# PROPER SPLIT: Train 60%, Val 20%, Test 20%
print("\n[3/6] Creating train/val/test split...")
X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
    X_raw, y, test_size=0.4, random_state=42, stratify=y
)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp_raw, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

X_train_summ, X_temp_summ = train_test_split(
    X_summ, test_size=0.4, random_state=42, stratify=y
)[0:2]
X_val_summ, X_test_summ = train_test_split(
    X_temp_summ, test_size=0.5, random_state=42, stratify=y_temp
)[0:2]

print(f"  Train: {len(y_train)} ({len(y_train)/len(y)*100:.0f}%)")
print(f"  Val:   {len(y_val)} ({len(y_val)/len(y)*100:.0f}%)")
print(f"  Test:  {len(y_test)} ({len(y_test)/len(y)*100:.0f}%)")

# Train diverse models
print("\n[4/6] Training diverse models...")

# Model 1: Random Forest (tree-based)
rf_raw = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_raw.fit(X_train_raw, y_train)
print("  ✅ Random Forest (Raw)")

# Model 2: Gradient Boosting (tree-based, different algorithm)
gb_raw = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
gb_raw.fit(X_train_raw, y_train)
print("  ✅ Gradient Boosting (Raw)")

# Model 3: MLP (neural network - DIFFERENT from trees!)
mlp_raw = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
mlp_raw.fit(X_train_raw, y_train)
print("  ✅ MLP Neural Network (Raw)")

# Model 4: RF on summary embeddings
rf_summ = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_summ.fit(X_train_summ, y_train)
print("  ✅ Random Forest (Summary)")

# Find optimal threshold on VALIDATION set (not test!)
print("\n[5/6] Finding optimal threshold on VALIDATION set...")

def find_optimal_threshold(model, X_val, y_val):
    pred_proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]

threshold_rf = find_optimal_threshold(rf_raw, X_val_raw, y_val)
threshold_gb = find_optimal_threshold(gb_raw, X_val_raw, y_val)
threshold_mlp = find_optimal_threshold(mlp_raw, X_val_raw, y_val)

print(f"  RF threshold: {threshold_rf:.3f}")
print(f"  GB threshold: {threshold_gb:.3f}")
print(f"  MLP threshold: {threshold_mlp:.3f}")

# Test set evaluation with validation-tuned thresholds
print("\n[6/6] Evaluating on TEST set...")

def evaluate_model(model, X_test, y_test, threshold, name):
    pred_proba = model.predict_proba(X_test)[:, 1]
    pred = (pred_proba >= threshold).astype(int)
    
    return {
        'Model': name,
        'AUROC': roc_auc_score(y_test, pred_proba),
        'F1': f1_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'Threshold': threshold
    }

results = []
results.append(evaluate_model(rf_raw, X_test_raw, y_test, threshold_rf, "RF Raw"))
results.append(evaluate_model(gb_raw, X_test_raw, y_test, threshold_gb, "GB Raw"))
results.append(evaluate_model(mlp_raw, X_test_raw, y_test, threshold_mlp, "MLP Raw"))
results.append(evaluate_model(rf_summ, X_test_summ, y_test, threshold_rf, "RF Summary"))

# Ensemble predictions
pred_rf = rf_raw.predict_proba(X_test_raw)[:, 1]
pred_gb = gb_raw.predict_proba(X_test_raw)[:, 1]
pred_mlp = mlp_raw.predict_proba(X_test_raw)[:, 1]
pred_rf_summ = rf_summ.predict_proba(X_test_summ)[:, 1]

# Simple average ensemble
pred_ensemble = (pred_rf + pred_gb + pred_mlp + pred_rf_summ) / 4

# Find ensemble threshold on validation
val_pred_rf = rf_raw.predict_proba(X_val_raw)[:, 1]
val_pred_gb = gb_raw.predict_proba(X_val_raw)[:, 1]
val_pred_mlp = mlp_raw.predict_proba(X_val_raw)[:, 1]
val_pred_rf_summ = rf_summ.predict_proba(X_val_summ)[:, 1]
val_pred_ensemble = (val_pred_rf + val_pred_gb + val_pred_mlp + val_pred_rf_summ) / 4

fpr, tpr, thresholds = roc_curve(y_val, val_pred_ensemble)
threshold_ens = thresholds[np.argmax(tpr - fpr)]

results.append({
    'Model': 'Ensemble (4 models)',
    'AUROC': roc_auc_score(y_test, pred_ensemble),
    'F1': f1_score(y_test, (pred_ensemble >= threshold_ens).astype(int)),
    'Precision': precision_score(y_test, (pred_ensemble >= threshold_ens).astype(int)),
    'Recall': recall_score(y_test, (pred_ensemble >= threshold_ens).astype(int)),
    'Threshold': threshold_ens
})

results_df = pd.DataFrame(results)

print("\n" + "="*70)
print("RESULTS (Proper Evaluation)")
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

rf_boot = bootstrap_f1(y_test, pred_rf, threshold_rf)
ens_boot = bootstrap_f1(y_test, pred_ensemble, threshold_ens)

print(f"\nRF Raw:   {rf_boot.mean():.4f} [95% CI: {np.percentile(rf_boot, 2.5):.4f}-{np.percentile(rf_boot, 97.5):.4f}]")
print(f"Ensemble: {ens_boot.mean():.4f} [95% CI: {np.percentile(ens_boot, 2.5):.4f}-{np.percentile(ens_boot, 97.5):.4f}]")

improvement = ens_boot - rf_boot
t_stat, p_value = stats.ttest_rel(ens_boot, rf_boot)

print(f"\nImprovement: {improvement.mean():.4f} ± {improvement.std():.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ Ensemble improvement IS statistically significant!")
else:
    print("⚠️  Ensemble improvement NOT statistically significant")

# Save results
results_df.to_csv('../results/proper_evaluation_results.csv', index=False)

print("\n✅ Results saved!")
print("\n" + "="*70)
print("KEY IMPROVEMENTS:")
print("="*70)
print("1. Proper train/val/test split (60%/20%/20%)")
print("2. Threshold tuning on validation set ONLY")
print("3. Added neural network for model diversity")
print("4. Bootstrap confidence intervals")
print("5. Statistical significance testing")
