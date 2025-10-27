# Critical Analysis of LLM-Based CRISPR Screen Prediction

**Hackathon Project: Improving Virtual CRISPR Screening with LLMs**

## 🎯 Project Overview

This project provides a critical analysis of the "Virtual CRISPR Screening" approach published in October 2024, which uses LLM embeddings to predict CRISPR screen results. Rather than attempting to beat their reported performance, we focus on identifying **fundamental limitations** and proposing **concrete improvements**.

## 📊 Key Findings

### Three Critical Biases Identified

1. **High-Confidence Exclusion Bias (95.4% data excluded)**
   - Only extreme effect genes included in benchmark
   - Ambiguous/borderline cases systematically excluded
   - Limits generalization to real-world applications

2. **Limited Test Set Diversity (2 papers, 2 cell lines)**
   - Benchmark from only 2 recent papers (Oct 2024)
   - 900 genes, 2 cell lines, 4 phenotypes
   - High variance, uncertain generalization

3. **Train/Test Distribution Mismatch (7.74% → 50%)**
   - Training: 7.74% positive (highly imbalanced)
   - Testing: 50% positive (balanced after inversion trick)
   - Real-world performance likely worse than reported

## 🔬 Our Approach

### Model Architecture
- **Embeddings**: Gene (3072) + Method (3072) + Cell (3072) + Phenotype (3072) = 12,288 dims
- **Classifiers**: Random Forest + Gradient Boosting
- **Ensemble**: Average predictions from multiple models

### Results
- **F1 Score**: 0.918 (vs paper's 0.84)
- **AUROC**: 0.933
- **FPR**: 0.092

*Note: Our higher performance is due to in-distribution testing (random split of benchmark) vs their out-of-distribution testing (new papers). This demonstrates that same-paper performance is easier than true generalization.*

## 💡 Proposed Improvements

### 1. ✅ Ensemble Methods (Implemented)
- **Status**: Complete
- **Impact**: +0.2% F1 improvement
- **Insight**: Limited improvement due to model similarity; more diversity needed

### 2. Context-Aware Embeddings
- **Current**: Static gene embeddings
- **Proposed**: Gene + cell + phenotype context in embedding generation
- **Expected Impact**: +5-10% F1

### 3. Biology-Specific Fine-Tuning
- **Current**: General-purpose text-embedding-3-large
- **Proposed**: Fine-tune on gene descriptions, pathways, CRISPR screens
- **Expected Impact**: +10-15% F1

### 4. Multi-Modal Integration
- **Current**: Text embeddings only
- **Proposed**: Combine text + expression + structure + networks
- **Expected Impact**: +15-20% F1

### 5. Active Learning Pipeline
- **Current**: Train on all high-confidence hits
- **Proposed**: Iteratively select informative borderline cases
- **Expected Impact**: 50% reduction in labeling costs

### 6. Uncertainty Quantification
- **Current**: Point predictions
- **Proposed**: Prediction + confidence intervals
- **Expected Impact**: Better trustworthiness and decision-making

## 📁 Repository Structure
```
embedding/analysis/
├── README.md                          # This file
├── code/
│   ├── run_full_analysis.py          # Main model training & evaluation
│   ├── bias_analysis.py              # Identify data biases
│   ├── error_analysis.py             # Analyze prediction failures
│   ├── ensemble_analysis.py          # Understand ensemble benefits
│   └── llm_improvements_analysis.py  # LLM-specific recommendations
├── data/
│   └── benchmark_raw.csv             # Test benchmark (1,814 examples)
├── results/
│   ├── model_comparison.csv          # Performance metrics
│   ├── predictions.csv               # Model predictions for analysis
│   └── llm_improvements.csv          # Improvement roadmap
└── plots/
    ├── roc_curves.png                # ROC curve comparison
    ├── f1_comparison.png             # F1 score comparison
    ├── bias_high_confidence_exclusion.png
    ├── bias_test_diversity.png
    ├── bias_class_distribution.png
    ├── error_patterns.png
    ├── ensemble_analysis.png
    └── llm_improvements_priority.png
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
module load anaconda3_gpu
source activate crispr
cd embedding/analysis/code
```

### 2. Run Complete Analysis
```bash
# Train models and generate predictions
python run_full_analysis.py

# Analyze biases
python bias_analysis.py

# Analyze errors
python error_analysis.py

# Analyze ensemble benefits
python ensemble_analysis.py

# Generate LLM improvement proposals
python llm_improvements_analysis.py
```

### 3. View Results
All plots saved to `../plots/`
All metrics saved to `../results/`

## 📈 Visualizations

### Bias Analysis
- **High-Confidence Exclusion**: Shows 95.4% of screened genes excluded
- **Test Diversity**: Limited to 2 cell lines and 4 phenotypes
- **Class Distribution**: Training (7.74% pos) vs Test (50% pos) mismatch

### Error Analysis
- Performance by cell line (glioblastoma harder than lung cancer)
- Performance by phenotype (PD1 blockade more variable)
- Misclassified gene examples

### Ensemble Analysis
- F1/AUROC comparison across models
- Theoretical benefits of ensembling
- Recommendations for increasing diversity

### LLM Improvements
- Priority ranking of 5 proposed improvements
- Implementation complexity vs expected impact
- Phased roadmap (quick wins → long-term)

## 🔍 Key Insights

1. **Bias Matters More Than Performance**
   - 95% data exclusion severely limits applicability
   - Training on extreme cases doesn't generalize to borderline effects
   - Need more diverse, representative benchmarks

2. **Ensemble Helps, But Limited**
   - Only +0.2% improvement due to model similarity
   - Need more diverse architectures (neural nets, different embeddings)
   - Still valuable for robustness

3. **LLM Embeddings Are Static**
   - Same gene embedding regardless of biological context
   - Context-aware generation could significantly improve accuracy
   - Biology-specific fine-tuning is critical next step

4. **Test Set Too Small**
   - 2 papers insufficient for reliable generalization estimates
   - High variance in performance metrics
   - Need validation on diverse prospective screens

## 🎓 Recommendations for Future Work

### Immediate (1-2 weeks)
1. Implement uncertainty quantification (ensemble disagreement)
2. Test context-aware embedding generation
3. Evaluate on additional cell lines

### Medium-term (1-2 months)
4. Fine-tune LLM on biological corpus
5. Implement active learning for borderline cases
6. Create diverse benchmark from multiple sources

### Long-term (2-3 months)
7. Multi-modal integration (text + expression + structure)
8. Prospective validation on new screens
9. Production deployment with uncertainty estimates

## 📚 References

1. Original Paper: "Predicting CRISPR screen results with LLM embeddings" (Oct 2024)
2. Benchmark: 2 papers, 1,814 gene-phenotype pairs, mouse screens

## 👥 Authors

**Hackathon Team**
- Critical analysis of bias and limitations
- Ensemble implementation and evaluation
- LLM-specific improvement proposals

## 📝 License

Educational project for hackathon purposes.

---

**Summary**: This analysis identifies three critical biases in LLM-based CRISPR prediction (95% data exclusion, limited test diversity, train/test mismatch) and proposes six concrete improvements, with ensemble methods already demonstrated. The work emphasizes understanding limitations over chasing performance metrics.

## ⚠️ Limitations & Honest Assessment

### Our Approach vs. Paper's Approach

**Key Differences:**

1. **Test Set**
   - **Paper**: Tests on completely new papers (out-of-distribution)
   - **Ours**: Random 70/30 split of their benchmark (in-distribution)
   - **Impact**: Our test is easier, explaining higher F1 (0.918 vs 0.84)

2. **Model Complexity**
   - **Paper**: 100M parameter MLP trained on 22.5M examples
   - **Ours**: Random Forest (~300K parameters) trained on 1,269 examples
   - **Impact**: Simpler model, faster training, sufficient for analysis purposes

3. **Scope**
   - **Paper**: Full training pipeline on 1,673 papers
   - **Ours**: Evaluation and analysis on their test benchmark
   - **Impact**: We replicate their *evaluation methodology*, not full training

### What This Means

✅ **Our analysis remains valid because:**
- Data biases we identify (95% exclusion, class imbalance) exist independently of model choice
- Error patterns reveal real biological challenges in the data
- Ensemble benefits demonstrate general ML principles
- LLM improvement proposals address fundamental approach limitations

❌ **What we don't claim:**
- Our model outperforms theirs (different test methodology)
- Our approach scales to full training dataset (computational constraints)
- Our F1 score represents out-of-distribution performance (in-distribution split)

### Why This Approach Serves the Hackathon Goal

The hackathon asked us to **"understand gaps and propose improvements"** in LLM-based CRISPR prediction, not to achieve state-of-the-art performance. Our focused analysis enables:

1. ✅ Identification of critical data biases (quantified at 95.4%, 7.74%, etc.)
2. ✅ Empirical testing of ensemble strategies (+0.2% improvement)
3. ✅ Systematic error pattern analysis
4. ✅ Concrete, actionable LLM-specific improvement proposals

**These insights are methodology-independent and directly address the core challenge.**

