# Triple Wise Attention Implementation (TWA)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-Required-green)](https://developer.nvidia.com/cuda-toolkit)

A research implementation that validates the performance of **triple-wise attention** against standard attention mechanisms through slot-based video reconstruction. This project provides a clean framework for comparing attention mechanisms and measuring their effectiveness in video processing tasks.

## 🎯 Overview

This project implements a novel **triple-wise attention mechanism** that splits Key (K) and Value (V) matrices into dual components (K1, K2, V1, V2), then merges them using element-wise multiplication. The performance is validated through a slot-based video reconstruction task using the [OpenVid-1k](https://huggingface.co/datasets/ACIDE/OpenVid-1k) dataset.

### Key Research Questions
- Does triple-wise attention improve video reconstruction quality?
- How does it compare to standard attention in terms of training efficiency?
- What is the computational overhead of the new mechanism?

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

# Run training with triple-wise attention (default)
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

## 🔬 Triple-wise Attention Mechanism

### Architecture

The triple-wise attention mechanism enhances standard attention by splitting K and V into dual components:

```python
# Standard Attention: Q, K, V
attention_output = softmax(Q @ K.T) @ V

# Triple-wise Attention: Q, K1, K2, V1, V2
k_merged = torch.cat([k1, k2], dim=-1)  # Concatenate K1 and K2
v_merged = torch.cat([v1, v2], dim=-1)  # Concatenate V1 and V2
attention_output = softmax(Q @ k_merged.T) @ v_merged
```

### Research Hypothesis

This approach enables:
- **Enhanced Representation Learning**: More complex attention patterns
- **Improved Reconstruction Quality**: Better video reconstruction
- **Increased Model Expressiveness**: Richer feature interactions

### Implementation Details

- **Parameter Overhead**: ~0% (17.5M parameters for both mechanisms)
- **FlashAttention Integration**: Optimized attention computation
- **Configurable**: Easy switching between standard and triple-wise attention
- **Concatenation Strategy**: K1,K2 and V1,V2 are concatenated along head dimension

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
'use_triple_wise_attention': True,  # Enable triple-wise attention

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
# Test Triple-wise Attention
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

## 📈 Expected Results

### Model Parameters
| Attention Type | Parameters | Overhead |
|----------------|------------|----------|
| Standard | ~17.5M | - |
| Triple-wise | ~17.5M | ~0% |

### Performance Metrics
- **Reconstruction Loss**: Primary quality metric
- **Training Stability**: Convergence behavior
- **Convergence Speed**: Epochs to optimal performance

### Success Criteria

Triple-wise attention is considered successful if it demonstrates:
- ✅ **Better Reconstruction Quality**: Lower final reconstruction loss
- ✅ **Stable Training**: Smooth convergence without instability
- ✅ **Efficient Implementation**: No significant parameter/compute overhead

## 🧪 Testing

```bash
# Run comprehensive tests
python test_twa.py
```

Tests include:
- Package structure validation
- Configuration loading
- Model instantiation
- Triple-wise attention functionality
- CUDA compatibility

## 🔧 Key Components

### Models
- **`SlotBasedVideoModel`**: Main model combining encoder and decoder
- **`SlotEncoder`**: Encodes tokens into slot representations
- **`SlotDecoder`**: Decodes slots back to token representations
- **`TripleWiseAttention`**: Novel attention implementation

### Attention Mechanisms
- **`Sinusoidal2DPositionEmbed`**: 2D positional encoding
- **`TripleWiseAttention`**: Dual-component attention mechanism

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

## 🎯 Research Focus

This project serves as a **validation platform** for triple-wise attention research:

1. **Clean Comparison**: Direct performance comparison between attention types
2. **Simplified Evaluation**: Single reconstruction loss for clear metrics
3. **Reproducible Results**: Standardized training and evaluation pipeline
4. **Extensible Framework**: Easy to modify for other attention mechanisms

## 📝 Notes

- **FlashAttention Requirement**: CUDA and fp16/bf16 precision needed
- **Single Loss Function**: Only reconstruction loss for clear evaluation
- **Research-Oriented**: Designed for attention mechanism validation
- **Well-Documented**: Clean code with English comments

---

**This project provides a comprehensive framework for validating novel attention mechanisms through video reconstruction tasks.**