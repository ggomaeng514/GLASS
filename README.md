# GLASS: Graph-aware, Label-aligned, Amortized Subset Selection

Official implementation of **"Explaining Black-Box Language Models: Learning to Optimize Linguistically-Structured Word Subsets"**.

## 🎯 Overview

GLASS is a novel method for explaining black-box deep language models (DLMs) by selecting small, informative subsets of input words. Unlike existing approaches, GLASS simultaneously achieves:

- ⚡ **Inference-time efficiency** through amortized optimization
- 🔒 **True black-box compatibility** without requiring gradients or internal access
- 🧠 **Linguistically coherent explanations** guided by syntactic/semantic structure

## 🏗️ Architecture

<img width="1500" height="509" alt="method" src="https://github.com/user-attachments/assets/f60631c0-ee05-450f-8d98-de5374cd747f" />


GLASS consists of three main components:

1. **Embedding Model** (f): Extracts contextualized word representations (frozen pre-trained model)
2. **GNN Selector** (πθ): Computes selection probabilities conditioned on linguistic structure
3. **Policy Gradient Training**: Learns discrete word selection without gradient access to the black-box model


## 📝 Key Contributions

1. **Novel Framework**: First method to simultaneously achieve efficiency, black-box compatibility, and linguistic coherence
2. **Policy Gradient Approach**: Gradient-free training for discrete word selection
3. **Structural Guidance**: Integration of linguistic graphs (syntax/semantics) for interpretable explanations
4. **Comprehensive Evaluation**: Extensive experiments across multiple architectures and tasks


## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/anonymous/GLASS.git
cd GLASS

# Create conda environment
conda create -n glass python=3.92
conda activate glass

# Install dependencies
pip install -r requirements.txt

# Install SpaCy language model for dependency parsing
python -m spacy download en_core_web_sm
```
## 📊 Experimental Results

### Performance Comparison (10% word selection)

| Method | Movies AUROC/AUPRC | Graph-SST2 AUROC/AUPRC | HateXplain AUROC/AUPRC |
|--------|---------------------|------------------------|------------------------|
| SHAP | 0.505 / 0.518 | 0.651 / 0.631 | 0.710 / 0.563 |
| LIME | 0.531 / 0.511 | 0.611 / 0.588 | 0.746 / 0.603 |
| IG | 0.544 / 0.501 | 0.581 / 0.570 | 0.761 / 0.613 |
| L2X | 0.596 / 0.590 | 0.552 / 0.558 | 0.624 / 0.479 |
| **GLASS** | **0.954 / 0.962** | **0.864 / 0.860** | **0.833 / 0.713** |
| Oracle (Full text) | 0.942 / 0.944 | 0.987 / 0.996 | 0.847 / 0.740 |

*GLASS achieves 96.5% of oracle performance using only 10% of input words!*

### Efficiency Comparison

| Method | Time per Sample | Peak GPU Memory |
|--------|-----------------|-----------------|
| KernelSHAP | 2.00 s | 7.35 GB |
| LIME | 6.13 s | 7.39 GB |
| IG | 0.67 s | 28.25 GB |
| **GLASS** | **0.08 s** | **6.24 GB** |

**GLASS is 8-77× faster than memoryless methods!**





### Linguistic Coherence: Alignment with Human Cognition

![structure_coherence](https://github.com/user-attachments/assets/ac117219-93ca-4a95-8ba0-b7ac2bd4c345)

GCN (w/ Syn) produces more coherent explanations by:
- Selecting grammatically central words (higher average degree)
- Forming fewer, larger word clusters (fewer subgraphs)
- Maintaining stronger connectivity (higher edge density)

This demonstrates that **incorporating syntactic structure** yields explanations that better match human linguistic intuition.

### Explanation Examples

<img width="4336" height="672" alt="example_gcn" src="https://github.com/user-attachments/assets/e02199cd-280d-4397-b5da-212c0e7bcbeb" />

**Left: Text annotations**
- 🟢 **Green**: Selected words matching human rationales
- 🔴 **Red**: Unselected human rationales  
- 🟠 **Orange**: Selected words beyond human annotations

**Right:** Dependency graphs showing structural connectivity of selected words

GLASS identifies coherent, structurally-connected word subsets that strongly align with human rationales while preserving linguistic structure.

---

## 📁 Repository Structure
```
GLASS/
├── README.md                      # This file
├── environment.yml                # Conda environment specification
├── dataset.py                     # Dataset loading and preprocessing
├── model.py                       # Selector model architecture (GNN/MLP)
├── learning.py                    # Training loop with policy gradient
├── learning_predictor.py          # Inference and explanation generation
├── main.py                        # Training script
├── main_predictor.py              # Inference script
├── main_fix_rate_inference.py     # Fixed selection rate inference
├── make_graph_pickle.py           # Preprocess dependency graphs
└── utill.py                        # Utility functions
```
