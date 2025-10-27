# Triple Wise Attention Implementation (TWA)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-Required-green)](https://developer.nvidia.com/cuda-toolkit)

A research implementation of **dual key-value attention** mechanism that investigates whether higher-order (k > 2) reasoning can be approximated efficiently without explicit tensorization. This project provides a comprehensive evaluation framework for comparing attention mechanisms through slot-based video reconstruction tasks.

## 🎯 Research Overview

This project implements a **dual key-value attention** design that employs two independent key-value streams (K1, V1) and (K2, V2) under a shared softmax normalization, inducing competitive coupling across semantic manifolds. The research evaluates this mechanism on slot-based autoregressive video reconstruction, comparing global pairwise, local pairwise, and dual-stream variants.

### Key Research Questions
- Can we extend attention mechanisms to capture higher-order dependencies efficiently?
- Does dual-KV attention provide meaningful improvements over standard pairwise attention?
- What are the computational trade-offs and capacity-dependent effectiveness patterns?

## 🚀 Quick Start

### Prerequisites
- Python >= 3.8
- CUDA-capable GPU (required for FlashAttention)
- Conda environment manager

### Installation

```bash
# Clone and navigate to project
cd triple_wise_attention_implementation

# Create and activate conda environment
conda create -n twa python=3.10
conda activate twa

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install project dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Run Training

```bash
# Activate environment
conda activate twa

# Run training with dual-KV attention (default)
python train.py

# Test the implementation
python test_twa.py
```

## 📊 Dataset

This project uses the **OpenVid-1k** dataset from [ACIDE/OpenVid-1k](https://huggingface.co/datasets/ACIDE/OpenVid-1k) on Hugging Face.

| Property | Value |
|----------|-------|
| **Source** | [ACIDE/OpenVid-1k](https://huggingface.co/datasets/ACIDE/OpenVid-1k) |
| **Size** | 1,000 video samples |
| **Format** | Parquet files with frame sequences |
| **Content** | Diverse videos (nature, people, objects, activities) |
| **Usage** | Slot-based video reconstruction benchmark |

## 🔬 Dual Key-Value Attention Mechanism

### Architecture

The dual key-value attention mechanism enhances standard attention by introducing two independent KV streams:

```python
# Standard Attention: Q, K, V
attention_output = softmax(Q @ K.T) @ V

# Dual-KV Attention: Q, K1, K2, V1, V2
K_merged = torch.cat([K1, K2], dim=-1)  # Concatenate K1 and K2
V_merged = torch.cat([V1, V2], dim=-1)  # Concatenate V1 and V2
attention_output = softmax(Q @ K_merged.T) @ V_merged
```

### Research Hypothesis

This approach enables:
- **Competitive Multi-Stream Attention**: Shared softmax creates competition between KV manifolds
- **Enhanced Representation Learning**: Dual streams provide richer semantic diversity
- **Improved Slot Separation**: Better disentanglement of object-level features

### Implementation Details

- **Parameter Overhead**: ~0% (same parameter count as standard attention)
- **Computational Overhead**: ~1.8× training time due to doubled KV sequence length
- **FlashAttention Integration**: Optimized attention computation
- **Configurable**: Easy switching between standard and dual-KV attention

## 🏗️ Project Structure

```
triple_wise_attention_implementation/
├── setup.py                    # Package configuration
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── test_twa.py                 # Test suite
└── twa/                        # Main package
    ├── __init__.py
    ├── config.py               # Configuration settings
    ├── train.py                # Training script
    └── modules/                # Core modules
        ├── __init__.py
        ├── modules.py          # Model implementations
        ├── rope.py             # Position embeddings
        └── utils.py            # Utilities
```

## 🎮 Usage

### Training Configuration

Edit `twa/config.py` to customize:

```python
# Attention mechanism selection
'use_triple_wise_attention': True,  # Enable dual-KV attention

# Model architecture
'slot_dim': 256,                   # Slot dimension
'token_dim': 256,                  # Token dimension
'num_slot_pool': 8,                # Number of slots

# Training settings
'batch_size': 1,                   # Batch size
'lr': 1e-4,                        # Learning rate
'num_epochs': 160,                 # Training epochs
```

### Compare Attention Mechanisms

```bash
# Test Dual-KV Attention
Config.use_triple_wise_attention = True
python train.py

# Test Standard Attention
Config.use_triple_wise_attention = False
python train.py
```

### Monitor Training

- **TensorBoard**: `runs/` directory for training logs
- **Checkpoints**: `checkpoints/` directory for model saves
- **Primary Metric**: Reconstruction loss (lower is better)

## 📈 Experimental Results

### Model Configurations

We systematically evaluate five configurations with varying capacity:

| Group | Heads (H) | Layers (L) | Capacity Description |
|-------|------------|------------|---------------------|
| G1 | 1 | 1 | Minimal (baseline) |
| G2 | 4 | 1 | Moderate single-layer |
| G3 | 8 | 1 | High-capacity single-layer |
| G4 | 8 | 4 | Deep multi-layer |
| G5 | 8 | 6 | Deepest configuration |

### Performance Results

#### Test Set MSE (Lower is Better)

| Configuration | Standard Attention | Dual-KV Attention | Improvement |
|---------------|-------------------|-------------------|-------------|
| G1 (1,1) | 0.551 | 0.399 | **-27.6%** |
| G2 (4,1) | 0.312 | 0.238 | **-23.7%** |
| G3 (8,1) | 0.225 | 0.175 | **-22.2%** |
| G4 (8,4) | 0.153 | 0.140 | **-8.5%** |
| G5 (8,6) | 0.113 | 0.102 | **-9.7%** |

#### Training Time Overhead

| Configuration | Standard Time | Dual-KV Time | Overhead |
|---------------|---------------|--------------|----------|
| G1 | 8.0 min | 14.4 min | **1.80×** |
| G2 | 9.7 min | 17.4 min | **1.80×** |
| G3 | 10.7 min | 19.2 min | **1.80×** |
| G4 | 42.6 min | 76.7 min | **1.80×** |
| G5 | 63.9 min | 115.0 min | **1.80×** |

### Key Findings

1. **Capacity-Dependent Effectiveness**: Dual-KV attention provides the most substantial benefits in low-capacity settings (G1-G3), with improvements diminishing as model capacity increases (G4-G5).

2. **Consistent Computational Overhead**: The 1.8× training time overhead is consistent across all configurations, reflecting the doubled KV sequence length.

3. **Diminishing Returns**: While achieving 20-27% improvement in low-capacity settings, the gain drops to under 10% for deep models despite paying the full computational cost.

## 🧪 Testing

```bash
# Run comprehensive tests
python test_twa.py
```

Tests include:
- Package structure validation
- Configuration loading
- Model instantiation
- Dual-KV attention functionality
- CUDA compatibility

## 🔧 Key Components

### Models
- **`SlotBasedVideoModel`**: Main model combining encoder and decoder
- **`SlotEncoder`**: Encodes tokens into slot representations
- **`SlotDecoder`**: Decodes slots back to token representations
- **`DualKVAttention`**: Dual key-value attention implementation

### Attention Mechanisms
- **`Sinusoidal2DPositionEmbed`**: 2D positional encoding
- **`DualKVAttention`**: Dual-stream attention mechanism

### Training Features
- Mixed precision training (AMP)
- Gradient clipping
- Learning rate scheduling
- TensorBoard logging

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | >=2.0 | Deep learning framework |
| transformers | Latest | Model utilities |
| einops | Latest | Tensor operations |
| flash-attn | Latest | Optimized attention |
| timm | Latest | Model architectures |
| matplotlib | Latest | Visualization |
| pandas | Latest | Data handling |
| easydict | Latest | Configuration |

## 🎯 Research Contributions

This project provides three key contributions:

1. **Theoretical Analysis**: Explains why standard attention is inherently pairwise and why this limits higher-order relational modeling.

2. **Dual-KV Design**: Proposes a computationally efficient approximation to triple-wise reasoning, combining expressivity and scalability.

3. **Empirical Evaluation**: Reveals both benefits and structural limitations, providing insights into capacity-efficiency trade-offs.

## ⚠️ Limitations and Considerations

### When Dual-KV Attention Works Best
- **Low-capacity models** (1-8 heads, 1-2 layers)
- **Resource-constrained scenarios**
- **Early training stages** (first few epochs)

### When Dual-KV Attention May Not Be Suitable
- **High-capacity models** (8+ heads, 4+ layers)
- **Long training regimes** (diminishing returns over time)
- **Production environments** where 1.8× overhead is prohibitive

### Theoretical Limitations
- **Remains Pairwise**: Despite empirical benefits, the kernel structure remains fundamentally pairwise
- **No True Higher-Order**: Does not achieve genuine triple-wise reasoning capabilities
- **Competition Bias**: Operates primarily as a pairwise competition mechanism

## 📝 Notes

- **Research-Oriented**: This implementation focuses on validating dual-KV attention through video reconstruction
- **Honest Reporting**: Results include both benefits and limitations for transparent evaluation
- **Controlled Experiments**: All comparisons use identical architectures, optimization, and training procedures
- **Short Training Regime**: 4-epoch evaluation captures early training dynamics

## 🔗 References

This implementation is based on the research paper:
> "From Pairwise to k-Wise in Practice: A Literature Review on Locality Priors and Higher-Order Expressivity in Transformers" by Puyue Wang and Songqi Guo.

Key references:
- Slot Attention: Object-Centric Learning with Slot Attention (Locatello et al., 2020)
- Higher-Order Graph Transformers (Zhou et al., 2024)
- Tensor Attention Training (Liang et al., 2024)

---

**This project provides a comprehensive framework for evaluating dual key-value attention mechanisms through controlled video reconstruction experiments.**