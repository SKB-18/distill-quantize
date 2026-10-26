"""
QuantizedLinear PyTorch Module.
Stores quantized integer weights and performs fast dequantized GEMM.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.granularity import quantize_per_channel, quantize_per_tensor
from quantize.numerics import dequantize_symmetric, dequantize_asymmetric
from quantize.calibration import calibrate_mse


class QuantizedLinear(nn.Module):
    """
    PyTorch Linear Layer storing quantized weights in INT8 or INT4 representations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 8,
        symmetric: bool = True,
        granularity: str = "per_channel"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.symmetric = symmetric
        self.granularity = granularity

        # Quantized weight buffer (out_features, in_features)
        dtype = torch.int8 if symmetric else torch.uint8
        self.register_buffer("qweight", torch.zeros((out_features, in_features), dtype=dtype))

        # Scale buffer: shape (out_features, 1) if per_channel, else (1,)
        if granularity == "per_channel":
            self.register_buffer("scale", torch.ones((out_features, 1), dtype=torch.float32))
            if not symmetric:
                self.register_buffer("zero_point", torch.zeros((out_features, 1), dtype=torch.float32))
            else:
                self.zero_point = None
        else:
            self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))
            if not symmetric:
                self.register_buffer("zero_point", torch.tensor(0.0, dtype=torch.float32))
            else:
                self.zero_point = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(
        cls,
        float_linear: nn.Linear,
        bits: int = 8,
        symmetric: bool = True,
        granularity: str = "per_channel",
        calibration: str = "minmax"
    ) -> "QuantizedLinear":
        """
        Constructs a QuantizedLinear module from a standard float nn.Linear layer.
        """
        layer = cls(
            in_features=float_linear.in_features,
            out_features=float_linear.out_features,
            bias=float_linear.bias is not None,
            bits=bits,
            symmetric=symmetric,
            granularity=granularity
        )

        weight = float_linear.weight.data.clone().float()

        # Optional MSE clipping calibration
        if calibration == "mse" and symmetric:
            calib_res = calibrate_mse(weight, bits=bits, symmetric=symmetric)
            weight = calib_res["clipped_tensor"]

        # Quantize weight
        if granularity == "per_channel":
            q, scale, zero_point = quantize_per_channel(weight, dim=0, bits=bits, symmetric=symmetric)
        else:
            q, scale, zero_point = quantize_per_tensor(weight, bits=bits, symmetric=symmetric)

        layer.qweight.copy_(q)
        layer.scale.copy_(scale.view(layer.scale.shape))
        if not symmetric and zero_point is not None:
            layer.zero_point.copy_(zero_point.view(layer.zero_point.shape))

        if float_linear.bias is not None:
            layer.bias.data.copy_(float_linear.bias.data.float())

        return layer

    def unpack_weights(self, target_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Dequantizes the integer weights into floating point representation.
        """
        if self.symmetric:
            return dequantize_symmetric(self.qweight, self.scale, target_dtype=target_dtype)
        else:
            return dequantize_asymmetric(self.qweight, self.scale, self.zero_point, target_dtype=target_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dynamic on-the-fly dequantization.
        """
        w_float = self.unpack_weights(target_dtype=x.dtype)
        return F.linear(x, w_float, self.bias)

    def memory_bytes(self) -> int:
        """
        Calculates the theoretical stored weight bytes (accounting for bitwidth and scales).
        """
        weight_elements = self.in_features * self.out_features
        weight_bytes = (weight_elements * self.bits) // 8
        scale_bytes = self.scale.numel() * 4
        zp_bytes = self.zero_point.numel() * 4 if self.zero_point is not None else 0
        bias_bytes = self.bias.numel() * 4 if self.bias is not None else 0
        return weight_bytes + scale_bytes + zp_bytes + bias_bytes

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, symmetric={self.symmetric}, granularity='{self.granularity}', "
            f"bias={self.bias is not None}"
        )
