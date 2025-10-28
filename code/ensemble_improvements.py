"""
Intelligent ensemble strategies to improve over single RF
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from scipy import stats

print("="*70)
print("ENSEMBLE IMPROVEMENT STRATEGIES")
print("="*70)

# Load benchmark
df = pd.read_csv("../data/benchmark_raw.csv")

# Load official embeddings
print("\n[1/5] Loading embeddings...")
embeddings = {}

embeddings['gliobl_pos'] = {
    'genes': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY/genes_mouse.npy", allow_pickle=True).item(),
    'cells': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY/cells.npy", allow_pickle=True).item(),
    'phenotypes': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY/phenotypes.npy", allow_pickle=True).item(),
    'methods': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY/methods.npy", allow_pickle=True).item(),
}

embeddings['gliobl_neg'] = {
    'genes': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY_FOR_INVERSE/genes_mouse.npy", allow_pickle=True).item(),
    'cells': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY_FOR_INVERSE/cells.npy", allow_pickle=True).item(),
    'phenotypes': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY_FOR_INVERSE/phenotypes.npy", allow_pickle=True).item(),
    'methods': np.load("/projects/bfqi/data_test_difficult/SCREEN_18_HITS_ONLY_FOR_INVERSE/methods.npy", allow_pickle=True).item(),
}

embeddings['lung_pos'] = {
    'genes': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY/genes_mouse.npy", allow_pickle=True).item(),
    'cells': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY/cells.npy", allow_pickle=True).item(),
    'phenotypes': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY/phenotypes.npy", allow_pickle=True).item(),
    'methods': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY/methods.npy", allow_pickle=True).item(),
}

embeddings['lung_neg'] = {
    'genes': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY_FOR_INVERSE/genes_mouse.npy", allow_pickle=True).item(),
    'cells': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY_FOR_INVERSE/cells.npy", allow_pickle=True).item(),
    'phenotypes': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY_FOR_INVERSE/phenotypes.npy", allow_pickle=True).item(),
    'methods': np.load("/projects/bfqi/data_test_difficult/SCREEN_17_HITS_ONLY_FOR_INVERSE/methods.npy", allow_pickle=True).item(),
}

# Prepare features
print("\n[2/5] Preparing features...")

def get_embedding_set(cell, phenotype):
    if 'glioblastoma' in cell.lower():
        if 'increased sensitivity' in phenotype:
            return embeddings['gliobl_pos']
        else:
            return embeddings['gliobl_neg']
    else:
        if 'increased resistance' in phenotype:
            return embeddings['lung_pos']
        else:
            return embeddings['lung_neg']

X, y = [], []

for idx, row in df.iterrows():
    gene = row['gene']
    cell = row['cell']
    method = row['perturbation'].capitalize()
    phenotype = row['phenotype']
    
    emb_set = get_embedding_set(cell, phenotype)
    
    if (gene in emb_set['genes'] and cell in emb_set['cells'] and 
        phenotype in emb_set['phenotypes'] and method in emb_set['methods']):
        
        X.append(np.concatenate([
            emb_set['genes'][gene],
            emb_set['methods'][method],
            emb_set['cells'][cell],
            emb_set['phenotypes'][phenotype]
        ]))
        y.append(row['hit'])

X = np.array(X)
y = np.array(y)

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

# Train diverse models
print("\n[3/5] Training diverse models...")

# Model 1: Random Forest (our best)
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Model 2: Random Forest with different hyperparams
rf2 = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, random_state=43, n_jobs=-1)
rf2.fit(X_train, y_train)

# Model 3: Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
gb.fit(X_train, y_train)

# Model 4: Logistic Regression (linear, very different!)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# Model 5: Better MLP
mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=1000, random_state=42, early_stopping=True)
mlp.fit(X_train, y_train)

print("  ✅ 5 diverse models trained")

# Get validation predictions for weight tuning
print("\n[4/5] Optimizing ensemble weights on validation...")

val_preds = {
    'rf': rf.predict_proba(X_val)[:, 1],
    'rf2': rf2.predict_proba(X_val)[:, 1],
    'gb': gb.predict_proba(X_val)[:, 1],
    'lr': lr.predict_proba(X_val)[:, 1],
    'mlp': mlp.predict_proba(X_val)[:, 1]
}

# Evaluate each model on validation
val_f1s = {}
for name, preds in val_preds.items():
    fpr, tpr, thresholds = roc_curve(y_val, preds)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]
    f1 = f1_score(y_val, (preds >= threshold).astype(int))
    val_f1s[name] = f1
    print(f"  {name.upper():5s}: F1 = {f1:.4f}")

# Strategy 1: Simple average
pred_simple = np.mean([val_preds['rf'], val_preds['rf2'], val_preds['gb'], val_preds['lr'], val_preds['mlp']], axis=0)

# Strategy 2: Weighted by validation F1
weights = np.array([val_f1s['rf'], val_f1s['rf2'], val_f1s['gb'], val_f1s['lr'], val_f1s['mlp']])
weights = weights / weights.sum()
pred_weighted = (weights[0] * val_preds['rf'] + 
                 weights[1] * val_preds['rf2'] + 
                 weights[2] * val_preds['gb'] + 
                 weights[3] * val_preds['lr'] + 
                 weights[4] * val_preds['mlp'])

# Strategy 3: Only strong models (F1 > 0.88)
strong_models = [name for name, f1 in val_f1s.items() if f1 > 0.88]
if len(strong_models) > 0:
    pred_strong = np.mean([val_preds[m] for m in strong_models], axis=0)
else:
    pred_strong = val_preds['rf']

# Strategy 4: Stacking (meta-learner)
val_stack = np.column_stack([val_preds['rf'], val_preds['rf2'], val_preds['gb'], val_preds['lr'], val_preds['mlp']])
meta_learner = LogisticRegression(random_state=42)
meta_learner.fit(val_stack, y_val)

print(f"\n  Weights (by validation F1): {[f'{w:.3f}' for w in weights]}")
print(f"  Strong models (F1>0.88): {strong_models}")

# Evaluate on test set
print("\n[5/5] Test set evaluation...")

test_preds = {
    'rf': rf.predict_proba(X_test)[:, 1],
    'rf2': rf2.predict_proba(X_test)[:, 1],
    'gb': gb.predict_proba(X_test)[:, 1],
    'lr': lr.predict_proba(X_test)[:, 1],
    'mlp': mlp.predict_proba(X_test)[:, 1]
}

# Test ensemble predictions
test_simple = np.mean([test_preds['rf'], test_preds['rf2'], test_preds['gb'], test_preds['lr'], test_preds['mlp']], axis=0)
test_weighted = (weights[0] * test_preds['rf'] + 
                 weights[1] * test_preds['rf2'] + 
                 weights[2] * test_preds['gb'] + 
                 weights[3] * test_preds['lr'] + 
                 weights[4] * test_preds['mlp'])
test_strong = np.mean([test_preds[m] for m in strong_models], axis=0) if len(strong_models) > 0 else test_preds['rf']

test_stack = np.column_stack([test_preds['rf'], test_preds['rf2'], test_preds['gb'], test_preds['lr'], test_preds['mlp']])
test_stacking = meta_learner.predict_proba(test_stack)[:, 1]

# Find optimal thresholds and evaluate
def evaluate_with_optimal_threshold(y_true, y_pred_proba, name):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    return {
        'Method': name,
        'F1': f1_score(y_true, y_pred),
        'AUROC': roc_auc_score(y_true, y_pred_proba),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred)
    }

results = []
results.append(evaluate_with_optimal_threshold(y_test, test_preds['rf'], "RF (baseline)"))
results.append(evaluate_with_optimal_threshold(y_test, test_preds['rf2'], "RF2 (different params)"))
results.append(evaluate_with_optimal_threshold(y_test, test_preds['gb'], "Gradient Boosting"))
results.append(evaluate_with_optimal_threshold(y_test, test_preds['lr'], "Logistic Regression"))
results.append(evaluate_with_optimal_threshold(y_test, test_preds['mlp'], "MLP"))
results.append(evaluate_with_optimal_threshold(y_test, test_simple, "Ensemble: Simple Average"))
results.append(evaluate_with_optimal_threshold(y_test, test_weighted, "Ensemble: Weighted"))
results.append(evaluate_with_optimal_threshold(y_test, test_strong, f"Ensemble: Strong Only ({len(strong_models)} models)"))
results.append(evaluate_with_optimal_threshold(y_test, test_stacking, "Ensemble: Stacking"))

results_df = pd.DataFrame(results)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(results_df.to_string(index=False))

# Find best
best_idx = results_df['F1'].idxmax()
best_method = results_df.loc[best_idx, 'Method']
best_f1 = results_df.loc[best_idx, 'F1']
baseline_f1 = results_df.loc[0, 'F1']

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print(f"\n🏆 Best Method: {best_method}")
print(f"   F1 = {best_f1:.4f}")
print(f"\n📊 Improvement over RF baseline:")
print(f"   +{(best_f1 - baseline_f1)*100:.2f}% absolute")

if best_f1 > baseline_f1:
    print("\n✅ SUCCESS! Found ensemble strategy that beats single RF!")
else:
    print("\n⚠️  No ensemble beats RF alone - single model is best")

results_df.to_csv('../results/ensemble_strategies.csv', index=False)
print("\n✅ Results saved!")
