import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torchao.prototype.dtypes.uint2 import pack_uint2, unpack_uint2

from ._C import bitlinear158 as bitlinear158_cpp


def pack_int8_to_int2(input: torch.CharTensor) -> torch.CharTensor:
    assert (
        input.size(-1) % 4 == 0
    ), "Number of channels in last dimension should be divisible by 4."

    input0 = (input[..., 0::4] & 0b11) << 6
    input1 = (input[..., 1::4] & 0b11) << 4
    input2 = (input[..., 2::4] & 0b11) << 2
    input3 = (input[..., 3::4] & 0b11) << 0
    output = input0 + input1 + input2 + input3

    return output


class BitLinear158(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 8,
        eps: float = 1e-5,
        weight_only_quantization: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__()

        self.weight_only_quantization = weight_only_quantization

        weight = torch.empty((out_features, in_features), **factory_kwargs)
        weight = nn.Parameter(weight, requires_grad=True)
        self.register_parameter("weight", weight)

        if bias:
            bias = torch.empty((out_features,), **factory_kwargs)
            bias = nn.Parameter(bias, requires_grad=True)
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.in_features = in_features
        self.out_features = out_features

        self.bits = bits
        self.eps = eps

        self._reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of BitLinear158.

        Args:
            input (torch.Tensor): Tensor of shape (*batch_shape, in_features).

        Returns:
            torch.Tensor: Tensor of shape (*batch_shape, out_features).

        """
        weight = self.weight
        bias = self.bias
        bits = self.bits
        eps = self.eps
        weight_only_quantization = self.weight_only_quantization

        q = 2 ** (bits - 1)
        quantized_weight, scale = quantize_weight(weight, eps=eps)
        abs_input = torch.abs(input)
        gamma, _ = torch.max(abs_input, dim=-1, keepdim=True)
        gamma = torch.clamp(gamma, min=eps)
        quantized_input = input * q / gamma

        if weight_only_quantization:
            quantized_input = torch.clamp(quantized_input, min=-q, max=q - 1)
        else:
            quantized_input = round_clamp(quantized_input, min=-q, max=q - 1)

        x = F.linear(quantized_input, quantized_weight, bias=bias)
        output = x * (scale * gamma) / q

        return output

    def _reset_parameters(self) -> None:
        # https://github.com/pytorch/pytorch/blob/b66e3f0957b96b058c9b632ca60833d9717a9d8a/torch/nn/modules/linear.py#L106-L114
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


class BitLinear158Inference(ABC, nn.Module):
    """Base class of BitLinear158 for inference.

    Unlike ``BitLinear158``, quantization is performed during initialization.
    """

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of BitLinear158Inference.

        Args:
            input (torch.Tensor): Tensor of shape (*batch_shape, in_features).

        Returns:
            torch.Tensor: Tensor of shape (*batch_shape, out_features).

        """

    @classmethod
    def build_from_bitlinear158(cls, module: BitLinear158) -> "BitLinear158Inference":
        converted = cls(
            module.weight,
            bias=module.bias,
            bits=module.bits,
            eps=module.eps,
            weight_only_quantization=module.weight_only_quantization,
        )

        return converted


class BitLinear158Int8Inference(BitLinear158Inference):
    """BitLinear158 for inference containing int8 weights.

    Unlike ``BitLinear158``, quantization is performed during initialization.
    """

    def __init__(
        self,
        weight: nn.Parameter | torch.Tensor,
        bias: nn.Parameter | torch.Tensor | None = None,
        bits: int = 8,
        eps: float = 1e-5,
        weight_only_quantization: bool = False,
    ) -> None:
        super().__init__()

        quantized_weight, scale = quantize_weight(weight.data)

        if bias is not None:
            bias = bias.data

        assert quantized_weight.min().item() >= -1, "Min weight is less than -1."
        assert quantized_weight.max().item() <= 1, "Max weight is greater than -1."

        quantized_weight = quantized_weight.to(torch.int8)

        self.register_buffer("quantized_weight", quantized_weight)
        self.register_buffer("scale", scale)

        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias)

        self.out_features, self.in_features = weight.size()
        self.bits = bits
        self.eps = eps
        self.weight_only_quantization = weight_only_quantization

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of BitLinear158Inference.

        Args:
            input (torch.Tensor): Tensor of shape (*batch_shape, in_features).

        Returns:
            torch.Tensor: Tensor of shape (*batch_shape, out_features).

        """
        quantized_weight = self.quantized_weight
        scale = self.scale
        bias = self.bias
        bits = self.bits
        eps = self.eps
        weight_only_quantization = self.weight_only_quantization

        q = 2 ** (bits - 1)
        abs_input = torch.abs(input)
        gamma, _ = torch.max(abs_input, dim=-1, keepdim=True)
        gamma = torch.clamp(gamma, min=eps)
        quantized_input = input * q / gamma

        if weight_only_quantization:
            quantized_input = torch.clamp(quantized_input, min=-q, max=q - 1)
        else:
            quantized_input = round_clamp(quantized_input, min=-q, max=q - 1)

        quantized_weight = quantized_weight.to(quantized_input.dtype)
        x = F.linear(quantized_input, quantized_weight, bias=bias)
        output = x * (scale * gamma) / q

        return output


class BitLinear158Uint2Inference(BitLinear158Inference):
    """BitLinear158 for inference containing uint2 weights.

    Unlike ``BitLinear158``, quantization is performed during initialization.
    """

    def __init__(
        self,
        weight: nn.Parameter | torch.Tensor,
        bias: nn.Parameter | torch.Tensor | None = None,
        bits: int = 8,
        eps: float = 1e-5,
        weight_only_quantization: bool = False,
    ) -> None:
        super().__init__()

        quantized_weight, scale = quantize_weight(weight.data)

        if bias is not None:
            bias = bias.data

        assert quantized_weight.min().item() >= -1, "Min weight is less than -1."
        assert quantized_weight.max().item() <= 1, "Max weight is greater than -1."

        quantized_weight = quantized_weight.to(torch.int8)
        quantized_weight = pack_uint2(quantized_weight + 1)

        self.register_buffer("quantized_weight", quantized_weight)
        self.register_buffer("scale", scale)

        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias)

        self.out_features, self.in_features = weight.size()
        self.bits = bits
        self.eps = eps
        self.weight_only_quantization = weight_only_quantization

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of BitLinear158Inference.

        Args:
            input (torch.Tensor): Tensor of shape (*batch_shape, in_features).

        Returns:
            torch.Tensor: Tensor of shape (*batch_shape, out_features).

        """
        quantized_weight = self.quantized_weight
        scale = self.scale
        bias = self.bias
        bits = self.bits
        eps = self.eps
        weight_only_quantization = self.weight_only_quantization

        q = 2 ** (bits - 1)
        abs_input = torch.abs(input)
        gamma, _ = torch.max(abs_input, dim=-1, keepdim=True)
        gamma = torch.clamp(gamma, min=eps)
        quantized_input = input * q / gamma

        if weight_only_quantization:
            quantized_input = torch.clamp(quantized_input, min=-q, max=q - 1)
        else:
            quantized_input = round_clamp(quantized_input, min=-q, max=q - 1)

        quantized_weight = unpack_uint2(quantized_weight)
        quantized_weight = quantized_weight.to(quantized_input.dtype) - 1
        x = F.linear(quantized_input, quantized_weight, bias=bias)
        output = x * (scale * gamma) / q

        return output


class BitLinear158CppInference(BitLinear158Inference):
    """BitLinear158 for inference containing float32 weights.

    Unlike ``BitLinear158``, quantization is performed during initialization.
    """

    def __init__(
        self,
        weight: nn.Parameter | torch.Tensor,
        bias: nn.Parameter | torch.Tensor | None = None,
        bits: int = 8,
        eps: float = 1e-5,
        weight_only_quantization: bool = False,
    ) -> None:
        super().__init__()

        quantized_weight, scale = quantize_weight(weight.data)

        if bias is not None:
            bias = bias.data

        assert quantized_weight.min().item() >= -1, "Min weight is less than -1."
        assert quantized_weight.max().item() <= 1, "Max weight is greater than -1."

        quantized_weight = quantized_weight.to(torch.int8)

        self.register_buffer("quantized_weight", quantized_weight)
        self.register_buffer("scale", scale)

        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias)

        self.out_features, self.in_features = weight.size()
        self.bits = bits
        self.eps = eps
        self.weight_only_quantization = weight_only_quantization

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of BitLinear158Inference.

        Args:
            input (torch.Tensor): Tensor of shape (*batch_shape, in_features).

        Returns:
            torch.Tensor: Tensor of shape (*batch_shape, out_features).

        """
        quantized_weight = self.quantized_weight
        scale = self.scale
        bias = self.bias
        bits = self.bits
        eps = self.eps
        weight_only_quantization = self.weight_only_quantization

        q = 2 ** (bits - 1)
        abs_input = torch.abs(input)
        gamma, _ = torch.max(abs_input, dim=-1, keepdim=True)
        gamma = torch.clamp(gamma, min=eps)
        quantized_input = input * q / gamma

        if weight_only_quantization:
            quantized_input = torch.clamp(quantized_input, min=-q, max=q - 1)
        else:
            quantized_input = round_clamp(quantized_input, min=-q, max=q - 1)

        x = bitlinear158(quantized_input, quantized_weight, bias=bias)
        output = x * (scale * gamma) / q

        return output


class BitLinear158CppTraining(BitLinear158Inference):
    def __init__(
        self,
        weight: nn.Parameter | torch.Tensor,
        bias: nn.Parameter | torch.Tensor | None = None,
        bits: int = 8,
        eps: float = 1e-5,
        weight_only_quantization: bool = False,
    ) -> None:
        super().__init__()

        weight = nn.Parameter(weight.data, requires_grad=True)

        if bias is not None:
            bias = nn.Parameter(bias.data, requires_grad=True)

        self.register_parameter("weight", weight)

        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", bias)

        self.out_features, self.in_features = weight.size()
        self.bits = bits
        self.eps = eps
        self.weight_only_quantization = weight_only_quantization

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of BitLinear158.

        Args:
            input (torch.Tensor): Tensor of shape (*batch_shape, in_features).

        Returns:
            torch.Tensor: Tensor of shape (*batch_shape, out_features).

        """
        weight = self.weight
        bias = self.bias
        bits = self.bits
        eps = self.eps
        weight_only_quantization = self.weight_only_quantization

        q = 2 ** (bits - 1)
        quantized_weight, scale = quantize_weight(weight, eps=eps)
        abs_input = torch.abs(input)
        gamma, _ = torch.max(abs_input, dim=-1, keepdim=True)
        gamma = torch.clamp(gamma, min=eps)
        quantized_input = input * q / gamma

        if weight_only_quantization:
            quantized_input = torch.clamp(quantized_input, min=-q, max=q - 1)
        else:
            quantized_input = round_clamp(quantized_input, min=-q, max=q - 1)

        x = bitlinear158(quantized_input, quantized_weight, bias=bias)
        output = x * (scale * gamma) / q

        return output


if version.parse(torch.__version__) >= version.parse("2.4.0"):

    @torch.library.register_fake(
        "bitlinear158compression::bitlinear158_inference_forward"
    )
    def _(
        input: torch.Tensor,
        quantized_weight: torch.Tensor,
    ) -> torch.Tensor:
        torch._check(input.dim() > 1)
        torch._check(quantized_weight.dim() == 2)
        torch._check(input.size(-1) == quantized_weight.size(1))
        torch._check(input.device == quantized_weight.device)

        *batch_shape, _ = input.size()
        out_features = quantized_weight.size(0)

        return torch.empty((*batch_shape, out_features), dtype=input.dtype)

    def bitlinear158(
        input: torch.Tensor,
        quantized_weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if quantized_weight.is_cuda:
            quantized_weight = quantized_weight.to(torch.int8)
        else:
            quantized_weight = quantized_weight.to(input.dtype)

        output = (
            torch.ops.bitlinear158compression.bitlinear158_inference_forward.default(
                input, quantized_weight
            )
        )

        if bias is not None:
            output = output + bias

        return output

    def _backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input, quantized_weight = ctx.saved_tensors

        grad_inputs = (
            torch.ops.bitlinear158compression.bitlinear158_inference_backward.default(
                input, quantized_weight, grad_output
            )
        )

        if ctx.needs_input_grad[0]:
            grad_input = grad_inputs[0]
        else:
            grad_input = None

        if ctx.needs_input_grad[1]:
            grad_quantized_weight = grad_inputs[1]
        else:
            grad_quantized_weight = None

        return grad_input, grad_quantized_weight

    def _setup_context(
        ctx,
        inputs: tuple[torch.Tensor, torch.Tensor],
        output: tuple[torch.Tensor],
    ) -> None:
        ctx.save_for_backward(*inputs)

    torch.library.register_autograd(
        "bitlinear158compression::bitlinear158_inference_forward",
        _backward,
        setup_context=_setup_context,
    )

else:

    class BitLinear158CppFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx, input: torch.Tensor, quantized_weight: torch.Tensor
        ) -> torch.Tensor:
            if quantized_weight.is_cuda:
                quantized_weight = quantized_weight.to(torch.int8)
            else:
                quantized_weight = quantized_weight.to(input.dtype)

            (output,) = bitlinear158_cpp.forward(input, quantized_weight)

            ctx.save_for_backward(input, quantized_weight)

            return output

        @staticmethod
        def backward(
            ctx, grad_output: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
            input, quantized_weight = ctx.saved_tensors
            grad_input, grad_quantized_weight = bitlinear158_cpp.backward(
                input, quantized_weight, grad_output
            )

            return grad_input, grad_quantized_weight

    def bitlinear158(
        input: torch.Tensor,
        quantized_weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = BitLinear158CppFunction.apply(input, quantized_weight)

        if bias is not None:
            output = output + bias

        return output


def round_clamp(
    input: torch.Tensor,
    min: float | None = None,
    max: float | None = None,
) -> torch.Tensor:
    """Differntiable round + clamp used in BitNet.

    .. note::

        Gradient is given by straight through estimator.

    """
    kwargs = {}

    if min is not None:
        kwargs["min"] = min

    if max is not None:
        kwargs["max"] = max

    x = torch.round(input)

    if len(kwargs) > 0:
        x = torch.clamp(x, **kwargs)

    if torch.is_grad_enabled():
        output = x
    else:
        output = torch.detach(x - input) + input

    return output


def quantize_weight(
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    abs_weight = torch.abs(weight)
    scale = torch.mean(abs_weight)
    scale = torch.clamp(scale, min=eps)
    weight = weight / scale
    quantized_weight = round_clamp(weight, min=-1, max=1)

    return quantized_weight, scale
