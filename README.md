# GLASS: Graph-aware, Label-aligned, Amortized Subset Selection

Official implementation of **"Explaining Black-Box Language Models: Learning to Optimize Linguistically-Structured Word Subsets"**.

## 🎯 Overview

GLASS is a novel method for explaining black-box deep language models (DLMs) by selecting small, informative subsets of input words. Unlike existing approaches, GLASS simultaneously achieves:

- ⚡ **Inference-time efficiency** through amortized optimization
- 🔒 **True black-box compatibility** without requiring gradients or internal access
- 🧠 **Linguistically coherent explanations** guided by syntactic/semantic structure

## 🏗️ Architecture

<p align="center">
  <img src="assets/architecture.png" alt="GLASS Architecture" width="800"/>
</p>

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

| Method | Movies ACC | Movies AUROC | Graph-SST2 ACC | HateXplain ACC |
|--------|------------|--------------|----------------|----------------|
| LIME | 0.495 | 0.531 | 0.502 | 0.572 |
| SHAP | 0.491 | 0.505 | 0.511 | 0.586 |
| IG | 0.503 | 0.544 | 0.500 | 0.587 |
| L2X | 0.561 | 0.596 | 0.526 | 0.493 |
| **GLASS** | **0.829** | **0.954** | **0.767** | **0.673** |
| Oracle (Full) | 0.859 | 0.942 | 0.842 | 0.674 |

*GLASS achieves 96.5% of oracle performance using only 10% of input words!*

### Efficiency Comparison

| Method | Time per Sample | Peak GPU Memory |
|--------|-----------------|-----------------|
| LIME | 6.13 s | 7.386 GB |
| KernelSHAP | 2.00 s | 7.345 GB |
| IG | 0.67 s | 28.245 GB |
| **GLASS** | **0.08 s** | **6.235 GB** |

**GLASS is 8-77× faster than memoryless methods!**


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
