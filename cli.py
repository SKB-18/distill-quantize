"""
Command-Line Interface (CLI) for Distill-Quantize.
Provides subcommands for distillation training, model quantization, and comparative benchmarking.
"""

import argparse
import sys
import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from distill.losses import HintonKDLoss
from distill.trainer import DistillationTrainer
from layers.converter import convert_to_quantized
from eval.benchmark import BenchmarkRunner


class SimpleModel(nn.Module):
    def __init__(self, in_features=64, hidden_features=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def run_distill(args):
    print("🚀 Running Knowledge Distillation Demo...")
    torch.manual_seed(42)

    teacher = SimpleModel(in_features=32, hidden_features=256, num_classes=5)
    student = SimpleModel(in_features=32, hidden_features=64, num_classes=5)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    loss_fn = HintonKDLoss(temperature=args.temperature, alpha=args.alpha)
    trainer = DistillationTrainer(student, teacher, optimizer, kd_loss=loss_fn)

    # Synthetic batch
    inputs = torch.randn(32, 32)
    labels = torch.randint(0, 5, (32,))

    print(f"Distilling student ({sum(p.numel() for p in student.parameters())} params) "
          f"from teacher ({sum(p.numel() for p in teacher.parameters())} params)...")

    for step in range(args.steps):
        step_metrics = trainer.train_step(inputs, labels)
        if (step + 1) % 5 == 0 or step == 0:
            print(f"Step {step_metrics['step']}: loss = {step_metrics['total_loss']:.4f}")

    print("✅ Distillation training finished successfully.")


def run_quantize(args):
    print(f"⚡ Quantizing model to {args.bits}-bit ({'symmetric' if args.symmetric else 'asymmetric'})...")
    torch.manual_seed(42)
    model = SimpleModel(in_features=128, hidden_features=256, num_classes=10)

    q_model, stats = convert_to_quantized(
        model,
        bits=args.bits,
        symmetric=args.symmetric,
        granularity=args.granularity
    )

    print("\nQuantization Statistics:")
    print(f"- Converted Layers: {stats['converted_layers']}")
    print(f"- Original Size:    {stats['orig_size_mb']:.4f} MB")
    print(f"- Quantized Size:   {stats['quant_size_mb']:.4f} MB")
    print(f"- Compression:      {stats['compression_ratio']:.2f}x")
    print("✅ Quantization completed.")


def run_benchmark(args):
    print("📊 Executing Empirical Benchmark...")
    torch.manual_seed(42)
    model = SimpleModel(in_features=128, hidden_features=512, num_classes=20)
    sample_input = torch.randn(8, 128)

    runner = BenchmarkRunner(model, sample_input, num_timing_runs=10)
    results = runner.run_suite()

    base = results["baseline_fp32"]
    print("\nBenchmark Summary:")
    print(f"Baseline FP32: size={base['size_mb']:.2f}MB, latency={base['latency_ms']:.2f}ms")
    for name, v in results["variants"].items():
        print(f"{name:18}: ratio={v['compression_ratio']:.2f}x, SQNR={v['sqnr_db']:.2f}dB, Cosine={v['cosine_sim']:.4f}, latency={v['latency_ms']:.2f}ms")


def main():
    parser = argparse.ArgumentParser(description="Distill-Quantize CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Distill
    p_distill = subparsers.add_parser("distill", help="Run knowledge distillation")
    p_distill.add_argument("--steps", type=int, default=15, help="Number of training steps")
    p_distill.add_argument("--temperature", type=float, default=4.0, help="Softmax temperature")
    p_distill.add_argument("--alpha", type=float, default=0.5, help="Loss interpolation alpha")
    p_distill.set_defaults(func=run_distill)

    # Quantize
    p_quant = subparsers.add_parser("quantize", help="Quantize a model")
    p_quant.add_argument("--bits", type=int, default=8, choices=[4, 8], help="Bitwidth (4 or 8)")
    p_quant.add_argument("--symmetric", type=lambda x: (str(x).lower() == 'true'), default=True, help="Symmetric mode")
    p_quant.add_argument("--granularity", type=str, default="per_channel", choices=["per_channel", "per_tensor"], help="Granularity")
    p_quant.set_defaults(func=run_quantize)

    # Benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run comparative benchmark")
    p_bench.set_defaults(func=run_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
