"""
Comparative Model Benchmark Runner:
Evaluates memory footprints, distortion (SQNR/Cosine), and forward pass latency.
"""

from typing import Dict, Any, List
import time
import copy
import torch
import torch.nn as nn
from layers.converter import convert_to_quantized, get_model_size_bytes
from .metrics import evaluate_quantization_distortion


class BenchmarkRunner:
    """
    Evaluates baseline FP32 model against multiple quantization configurations.
    """

    def __init__(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        num_timing_runs: int = 15,
        warmup_runs: int = 3
    ):
        self.model = model.eval()
        self.sample_input = sample_input
        self.num_timing_runs = num_timing_runs
        self.warmup_runs = warmup_runs

    def _measure_latency(self, model: nn.Module) -> float:
        """Measures median latency per forward pass in milliseconds."""
        model.eval()
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(self.sample_input)

            latencies = []
            for _ in range(self.num_timing_runs):
                t0 = time.perf_counter()
                _ = model(self.sample_input)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

        latencies.sort()
        return latencies[len(latencies) // 2]  # median latency

    def run_suite(self) -> Dict[str, Any]:
        """
        Runs comprehensive benchmark suite across FP32, INT8-Symmetric, INT8-Asymmetric, and INT4.
        """
        # Baseline FP32
        fp32_size = get_model_size_bytes(self.model) / (1024 * 1024)
        fp32_latency = self._measure_latency(self.model)

        with torch.no_grad():
            fp32_out = self.model(self.sample_input)
            if isinstance(fp32_out, (tuple, list)):
                fp32_out = fp32_out[0]

        results = {
            "baseline_fp32": {
                "size_mb": fp32_size,
                "latency_ms": fp32_latency,
                "compression_ratio": 1.0,
                "sqnr_db": float("inf"),
                "cosine_sim": 1.0
            },
            "variants": {}
        }

        configurations = [
            ("int8_symmetric", 8, True, "per_channel"),
            ("int8_asymmetric", 8, False, "per_channel"),
            ("int4_symmetric", 4, True, "per_channel"),
        ]

        for name, bits, symmetric, granularity in configurations:
            q_model, stats = convert_to_quantized(
                self.model,
                bits=bits,
                symmetric=symmetric,
                granularity=granularity,
                inplace=False
            )

            with torch.no_grad():
                q_out = q_model(self.sample_input)
                if isinstance(q_out, (tuple, list)):
                    q_out = q_out[0]

            distort = evaluate_quantization_distortion(fp32_out, q_out)
            latency = self._measure_latency(q_model)

            results["variants"][name] = {
                "size_mb": stats["quant_size_mb"],
                "compression_ratio": stats["compression_ratio"],
                "latency_ms": latency,
                "sqnr_db": distort["sqnr_db"],
                "cosine_sim": distort["cosine_similarity"],
                "mae": distort["mae"]
            }

        return results
