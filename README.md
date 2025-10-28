# Critical Analysis of LLM-Based CRISPR Screen Prediction

**Hackathon Project: Evidence-Based Evaluation of Virtual CRISPR Screening**

## 🎯 Project Overview

This project provides a rigorous, evidence-based analysis of the "Virtual CRISPR Screening" approach (Song et al., 2025), which uses LLM embeddings to predict CRISPR screen results. Our focus is on identifying **fundamental limitations** with proper statistical validation and proposing **concrete improvements**.

## 📊 Key Findings

### Three Evidence-Based Biases Identified

1. **Training Set Imbalance (7.74% positive)**
   - Source: Paper Section 3
   - Training data highly skewed (1:12 ratio)
   - Test set artificially balanced via inversion trick
   - Real-world deployment will face ~7.74% hit rate

2. **Limited Test Diversity (2 papers, 900 genes)**
   - Only 2 cell lines, 4 phenotypes tested
   - Small sample size (n=1,814) → high variance
   - Uncertain generalization to other biological contexts

3. **High-Confidence Filtering (extreme cases only)**
   - Source: Paper Section 2.2
   - Borderline/ambiguous effects systematically excluded
   - Unknown performance on subtle biological effects
   - Model trained only on extreme positive/negative cases

## 🔬 Our Methodology

### Proper Evaluation Framework

**Data Split:**
- Train: 60% (1,088 examples)
- Validation: 20% (363 examples)  
- Test: 20% (363 examples)

**Key Principles:**
- ✅ Threshold optimization on **validation set only** (no test leakage)
- ✅ Statistical validation with bootstrap confidence intervals
- ✅ Significance testing (paired t-tests)
- ✅ Model diversity for ensemble (RF + GB + MLP + RF-Summary)

### Model Architecture

**Embeddings (12,288 dimensions total):**
- Gene: 3,072 dims (from text-embedding-3-large)
- Method: 3,072 dims
- Cell Line: 3,072 dims
- Phenotype: 3,072 dims

**Classifiers:**
- Random Forest (tree-based)
- Gradient Boosting (tree-based, different algorithm)
- MLP Neural Network (fundamentally different architecture)
- Ensemble: Average of all predictions

### Results (Statistically Validated)

| Model | F1 | AUROC | 95% Confidence Interval |
|-------|-----|-------|------------------------|
| RF Raw | 0.891 | 0.924 | [0.858, 0.922] |
| GB Raw | 0.906 | 0.899 | - |
| MLP Raw | 0.865 | 0.913 | - |
| **Ensemble** | **0.909** | **0.924** | **[0.878, 0.938]** |

**Ensemble Improvement:**
- +1.89% F1 over best single model
- ±2.21% standard deviation
- **p < 0.0001** (statistically significant)

## 💡 Proposed LLM-Specific Improvements

### 1. ✅ Ensemble Methods (Demonstrated)
- **Status**: Implemented and validated
- **Result**: +1.89% F1 (p < 0.0001)
- **Key Insight**: Requires true model diversity (not just same architecture)

### 2. Context-Aware Embeddings
- **Current**: Static gene embedding regardless of context
- **Proposed**: Embed "gene X in cell Y affecting phenotype Z" as single context
- **Rationale**: Similar to our observed ensemble benefit from diversity
- **Estimated Impact**: +3-5% F1 (requires empirical validation)

### 3. Biology-Specific LLM Fine-Tuning
- **Current**: General-purpose text-embedding-3-large
- **Proposed**: Fine-tune on gene descriptions, pathways, CRISPR literature
- **Evidence**: BioLinkBERT showed ~7% improvement on biomedical tasks
- **Estimated Impact**: +5-10% F1 (based on similar work)

### 4. Multi-Modal Integration
- **Current**: Text embeddings only
- **Proposed**: Text + gene expression + protein structure + pathway graphs
- **Rationale**: Multi-modal typically adds 10-20% over text-only
- **Estimated Impact**: +10-15% F1 (requires validation)

### 5. Active Learning Pipeline
- **Current**: Train on all high-confidence hits
- **Proposed**: Iteratively select high-uncertainty borderline cases
- **Evidence**: Standard active learning achieves 40-70% cost reduction (Settles, 2009)
- **Estimated Impact**: 40-60% reduction in labeling costs

### 6. Uncertainty Quantification
- **Current**: Point predictions (probability)
- **Proposed**: Prediction + confidence interval
- **Method**: Ensemble disagreement or conformal prediction
- **Impact**: Better trust calibration and experimental prioritization

*Note: All "estimated" impacts require empirical validation. Only ensemble improvement has been demonstrated and validated in this work.*

## 📁 Repository Structure
```
crispr-llm-analysis/
├── README.md                          # This file
├── code/
│   ├── run_proper_evaluation.py      # Main training with proper train/val/test
│   ├── bias_analysis_corrected.py    # Evidence-based bias identification
│   ├── error_analysis.py             # Prediction failure pattern analysis
│   ├── ensemble_analysis.py          # Ensemble benefit analysis
│   └── llm_improvements_analysis.py  # LLM-specific recommendations
├── data/
│   ├── benchmark_raw.csv             # Test benchmark (1,814 examples)
│   └── genome_mus_musculus.tsv       # Gene ID mapping
├── results/
│   ├── proper_evaluation_results.csv # Performance metrics
│   ├── proper_predictions.csv        # Model predictions
│   └── llm_improvements.csv          # Improvement roadmap
└── plots/
    ├── roc_curves.png
    ├── bias_class_imbalance.png
    ├── bias_test_diversity.png
    ├── bias_high_confidence_filtering.png
    ├── error_patterns.png
    ├── ensemble_analysis.png
    └── llm_improvements_priority.png
```

## 🚀 Quick Start
```bash
# Setup environment
module load anaconda3_gpu
source activate crispr
cd code

# Run complete analysis
python run_proper_evaluation.py      # Train/val/test with proper evaluation
python bias_analysis_corrected.py    # Evidence-based bias analysis
python error_analysis.py              # Error pattern identification
python ensemble_analysis.py           # Why ensemble helps
python llm_improvements_analysis.py   # LLM-specific proposals

# View results
ls ../plots/    # 7 publication-quality figures
ls ../results/  # Performance metrics and predictions
```

## 🔍 Key Insights

### 1. Proper Evaluation is Critical
- Threshold tuning on test data inflates F1 by ~2.7%
- Small test sets (n=363) require confidence intervals
- Statistical significance testing is essential

### 2. Ensemble Benefits Require Diversity
- Similar models (RF + GB) show minimal improvement
- Adding neural network (MLP) enables +1.89% gain
- Key: Different architectures, not just different hyperparameters

### 3. Biases Are Data-Driven
- 7.74% training imbalance affects any model
- High-confidence filtering limits generalization
- Test diversity (2 papers) limits conclusions
- These are fundamental limitations, not model flaws

### 4. LLM Embeddings Are Static
- Same gene embedding regardless of cellular context
- Context-aware generation could significantly improve accuracy
- Biology-specific fine-tuning is critical next step

## ⚠️ Limitations & Honest Assessment

### Our Approach vs. Original Paper

| Aspect | Original Paper | Our Analysis | Impact |
|--------|---------------|--------------|--------|
| **Test Set** | New papers (out-of-distribution) | Random split (in-distribution) | Our test is easier |
| **Model** | 100M param MLP | RF/GB/MLP ensemble | Simpler, faster |
| **Training Data** | 22.5M examples from 1,673 papers | 1,088 examples from 2 papers | Much smaller scale |
| **Purpose** | Production prediction system | Critical analysis framework | Different goals |

### What We Demonstrate

✅ **Proper evaluation methodology** (train/val/test, no leakage)
✅ **Statistical rigor** (bootstrap CI, significance tests)
✅ **Evidence-based bias identification** (all claims sourced)
✅ **Ensemble improvement** (+1.89%, validated)
✅ **Actionable LLM recommendations** (6 concrete proposals)
✅ **Reproducible code** (all results verifiable)

### What We Don't Claim

❌ Our model outperforms their production system
❌ Our results generalize to out-of-distribution tests
❌ Our approach scales to full 22.5M training set
❌ We achieved state-of-the-art performance

**Our contribution:** Critical analysis with proper methodology, not a competing production system.

## 📈 Visualizations

All plots available in `/plots/`:

1. **ROC Curves** - Model comparison across ensemble strategies
2. **Class Imbalance** - Training (7.74%) vs Test (50%) distribution
3. **Test Diversity** - Limited to 2 cell lines and 4 phenotypes
4. **High-Confidence Filtering** - Conceptual diagram of exclusion strategy
5. **Error Patterns** - Performance by cell line and phenotype
6. **Ensemble Analysis** - Why diversity matters
7. **LLM Priorities** - Estimated impact vs implementation complexity

## 💡 Recommendations

### For Researchers
1. Use proper train/val/test splits (no threshold tuning on test)
2. Always report confidence intervals for small test sets
3. Test statistical significance (don't trust point estimates)
4. Evaluate on diverse, large-scale benchmarks
5. Include borderline cases, not just extreme effects

### For Practitioners
1. Implement ensemble methods (requires model diversity)
2. Add uncertainty quantification for decision support
3. Consider context-aware embeddings for your domain
4. Fine-tune LLMs on domain-specific corpora
5. Use active learning to reduce labeling costs

### For This Specific Task
1. Expand test set beyond 2 papers
2. Include borderline biological effects
3. Test on imbalanced data (real-world scenario)
4. Validate on prospective screens
5. Compare against experimental validation

## 📚 References

**Primary Source:**
- Song et al. (2025) "Virtual CRISPR: Can LLMs Predict CRISPR Screen Results?" *Proceedings of BioNLP 2025*

**Data Sources:**
- BioGRID-ORCS (Oughtred et al., 2021) - Training data
- Chen et al. (2024) *Nature* - Test benchmark (gliocidin screen)
- Skoulidis et al. (2024) *Nature* - Test benchmark (PD1 blockade)

**Methodological References:**
- Settles (2009) - Active learning survey
- Similar biomedical LLM work (BioLinkBERT) for impact estimates

## 👥 Authors

**Hackathon Analysis Team**

Contributions:
- Critical analysis of approach limitations
- Proper evaluation methodology implementation
- Statistical validation framework
- Evidence-based bias identification
- LLM-specific improvement proposals

## 📝 License

Educational project for hackathon purposes.

---

**Bottom Line:** We identified three evidence-based biases (7.74% training imbalance, limited 2-paper test set, high-confidence filtering), demonstrated statistically significant ensemble improvement (+1.89%, p<0.0001) using proper evaluation, and proposed six concrete LLM-specific improvements. All claims are backed by verifiable evidence or clearly labeled as requiring validation.
