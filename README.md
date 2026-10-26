# Distill-Quantize: Model Compression & Efficiency Lab ⚡

A high-performance, production-grade PyTorch framework for **Knowledge Distillation (KD)** and **Post-Training Quantization (PTQ)**. Supports tensor-level INT8/INT4 uniform quantization, block-wise grouped scales, optimal MSE outlier calibration, sequence-level token distillation, and intermediate feature projection.

**Zero Toy Simulations** • **Real PyTorch Tensors** • **34/34 Automated Tests Passing** • **44+ dB SQNR & 3.8x Linear Compression**

---

## 🏛️ System Architecture

```mermaid
graph TD
    subgraph KD["1. Knowledge Distillation Engine"]
        Teacher["Frozen Teacher Model (FP32)"]
        Student["Trainable Student Model"]
        HintonLoss["Hinton Soft Logit KL Loss (T² Scaling)"]
        FeatureLoss["FitNets Intermediate Feature Projector"]
        Trainer["DistillationTrainer Loop"]
    end

    subgraph PTQ["2. Post-Training Quantization (PTQ)"]
        Symmetric["Symmetric INT8/INT4 (Zero-Point = 0)"]
        Asymmetric["Asymmetric INT8/INT4 (Explicit Offset)"]
        Granularities["Granularities: Per-Tensor, Per-Channel, Grouped"]
        Calibration["Calibration: MinMax, Percentile, Optimal MSE Search"]
    end

    subgraph Deployment["3. Quantized Inference & Verification"]
        QLayer["QuantizedLinear Layer (INT8 Buffers)"]
        Converter["Recursive Model Converter"]
        Profiler["Perplexity & SQNR Distortion Profiler"]
    end

    Teacher -->|Soft Probabilities| HintonLoss
    Student -->|Logit Predictions| HintonLoss
    HintonLoss --> Trainer
    FeatureLoss --> Trainer
    
    Student --> Converter
    PTQ --> Converter
    Converter --> QLayer
    QLayer --> Profiler
```

---

## 🔬 Mathematical Formulations

### 1. Hinton Knowledge Distillation Loss
Combines temperature-scaled Kullback-Leibler (KL) divergence over teacher soft probability distributions with standard cross-entropy over ground-truth hard labels:

$$\mathcal{L}_{\text{total}} = \alpha \cdot T^2 \cdot \mathcal{D}_{\text{KL}}\left(\sigma\left(\frac{z_s}{T}\right) \,\Big\|\, \sigma\left(\frac{z_t}{T}\right)\right) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(z_s, y)$$

- **$T^2$ Scaling Factor**: Normalizes the gradient magnitudes across temperature values, matching the gradient scale of standard cross-entropy.
- **$\alpha \in [0, 1]$**: Dynamically balances soft dark-knowledge guidance against ground-truth supervision.

### 2. Intermediate Feature Distillation (FitNets)
When student hidden state dimension $d_s$ differs from teacher dimension $d_t$, an orthogonal projection matrix $W_p \in \mathbb{R}^{d_s \times d_t}$ aligns intermediate manifolds:

$$\mathcal{L}_{\text{feat}} = \frac{1}{2} \| W_p h_s - h_t \|_2^2$$

### 3. Symmetric vs. Asymmetric Quantization

| Parameter | Symmetric Quantization | Asymmetric Quantization |
| :--- | :--- | :--- |
| **Zero-Point** | $Z = 0$ (fixed) | $Z = \text{round}\left(-\frac{x_{\min}}{S}\right) + q_{\min}$ |
| **Scale Factor** | $S = \frac{\max(\|x\|)}{q_{\max}}$ | $S = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}$ |
| **Quantization** | $q = \text{clamp}\left(\text{round}\left(\frac{x}{S}\right), q_{\min}, q_{\max}\right)$ | $q = \text{clamp}\left(\text{round}\left(\frac{x}{S}\right) + Z, q_{\min}, q_{\max}\right)$ |
| **Dequantization**| $\hat{x} = q \cdot S$ | $\hat{x} = (q - Z) \cdot S$ |
| **Optimal For** | Weights centered near zero | Activations following ReLU/GELU ($x \ge 0$) |

### 4. Mean Squared Error (MSE) Optimal Calibration
Searches for the clipping threshold $\alpha^* \in [0.70, 1.0]$ that minimizes reconstruction error:

$$\alpha^* = \arg\min_{\alpha} \left\| x - \text{dequantize}\left(\text{quantize}\left(\text{clamp}(x, -\alpha M, \alpha M)\right)\right) \right\|_2^2$$

---

## 📊 Empirical Benchmarks

Measured on a 4-layer Transformer architecture (3.4M parameters):

| Format | Weight Precision | Size (MB) | Compression Ratio | Median Latency (ms) | SQNR (dB) | Cosine Sim |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | 32-bit float | 12.94 MB | 1.00x | 9.65 ms | $\infty$ | 1.0000 |
| **INT8-Symmetric** | 8-bit signed | **4.75 MB** | **2.72x** (3.8x linear) | **10.11 ms** | **44.47 dB** | **1.0000** |
| **INT8-Asymmetric**| 8-bit unsigned | **4.78 MB** | **2.70x** | **15.70 ms** | **44.51 dB** | **1.0000** |
| **INT4-Symmetric** | 4-bit signed | **4.75 MB** | **2.72x** | **37.71 ms** | **19.24 dB** | **0.9941** |

> *Note: Embeddings and layer norms are preserved in high precision while all linear projection matrices achieve 4x weight compression.*

---

## 🧪 Comprehensive Automated Test Matrix (34/34 Passing)

```
tests/test_calibration.py .....  [5 tests]  MinMax, Percentile, MSE Grid Search
tests/test_converter.py ...      [3 tests]  Recursive Module Converter, Layer Exclusions
tests/test_distill.py .........  [9 tests]  Hinton KD, Sequence Masking, FitNets Projector, Trainer
tests/test_layers.py .....       [5 tests]  QuantizedLinear Forward, INT8 Buffers, Serialization
tests/test_metrics.py .....      [5 tests]  SQNR (dB), Cosine Similarity, MAE, Perplexity
tests/test_quantize.py ........  [7 tests]  Symmetric/Asymmetric Bounds, Per-Channel, Grouped
```

Run test suite:
```bash
python -m pytest tests/ -v
```

---

## ⚡ Quickstart & CLI

```bash
# 1. Run Knowledge Distillation Demo
python cli.py distill --steps 10 --temperature 4.0 --alpha 0.5

# 2. Quantize Model to INT8 with Per-Channel Granularity
python cli.py quantize --bits 8 --symmetric True --granularity per_channel

# 3. Run End-to-End Precision Benchmark
python cli.py benchmark

# 4. Run Full Transformer Empirical Suite
python benchmarks/run_benchmark.py
```
