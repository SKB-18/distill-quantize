"""
Standalone Empirical Benchmark Suite for Distillation and Quantization.
Runs comparisons between FP32, INT8-Symmetric, INT8-Asymmetric, and INT4.
"""

import sys
import os
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval.benchmark import BenchmarkRunner


class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, d_ff=1024):
        super().__init__()
        self.attn = nn.Linear(d_model, d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.attn(x)
        x = self.norm(x + h)
        ff = self.ff2(self.act(self.ff1(x)))
        return self.norm(x + ff)


class BenchmarkModel(nn.Module):
    def __init__(self, vocab_size=1000, d_model=256, num_layers=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([SimpleTransformerBlock(d_model) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def main():
    print("=" * 70)
    print("🚀 Running Distill-Quantize Empirical Benchmark Suite")
    print("=" * 70)

    torch.manual_seed(42)
    model = BenchmarkModel(vocab_size=2000, d_model=256, num_layers=4)
    sample_input = torch.randint(0, 2000, (4, 64))

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Executing benchmark runs across precision formats...")

    runner = BenchmarkRunner(model, sample_input, num_timing_runs=10, warmup_runs=2)
    results = runner.run_suite()

    base = results["baseline_fp32"]
    print("\n" + "=" * 70)
    print("📊 BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Format':<18} | {'Size (MB)':<10} | {'Ratio':<8} | {'Latency (ms)':<12} | {'SQNR (dB)':<10} | {'Cosine Sim':<10}")
    print("-" * 78)
    print(f"{'FP32 Baseline':<18} | {base['size_mb']:<10.2f} | {base['compression_ratio']:<8.2f} | {base['latency_ms']:<12.2f} | {'inf':<10} | {base['cosine_sim']:<10.4f}")

    for name, v in results["variants"].items():
        print(f"{name:<18} | {v['size_mb']:<10.2f} | {v['compression_ratio']:<8.2f} | {v['latency_ms']:<12.2f} | {v['sqnr_db']:<10.2f} | {v['cosine_sim']:<10.4f}")

    print("=" * 70)
    print("✅ Benchmark complete.")


if __name__ == "__main__":
    main()
