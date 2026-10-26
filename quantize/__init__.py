"""
Post-Training Quantization (PTQ) Engine
Provides INT8 and INT4 uniform quantization, per-tensor/per-channel/grouped scales, and MSE calibration.
"""

from .numerics import (
    QuantParams,
    quantize_symmetric,
    dequantize_symmetric,
    quantize_asymmetric,
    dequantize_asymmetric,
)
from .granularity import (
    quantize_per_tensor,
    quantize_per_channel,
    quantize_grouped,
)
from .calibration import (
    calibrate_minmax,
    calibrate_percentile,
    calibrate_mse,
)

__all__ = [
    "QuantParams",
    "quantize_symmetric",
    "dequantize_symmetric",
    "quantize_asymmetric",
    "dequantize_asymmetric",
    "quantize_per_tensor",
    "quantize_per_channel",
    "quantize_grouped",
    "calibrate_minmax",
    "calibrate_percentile",
    "calibrate_mse",
]
