# BitLinear158Compression

In this library, we compare following compression models for inference by BitLinear158:

- `BitLinear158`: Ternary weight is managed by `float32` (naive implementation)
- `BitLinear158Int8Inference`: Ternary weight is managed by `int8`
- `BitLinear158Uint2Inference`: Ternary weight is managed by `uint2` defined in `torchao`
- `BitLinear158CppInference`: Ternary weight is managed by `int8` and forward pass is implemented by C++/CUDA. When weight is on CUDA, CUDA kernel is used under the hood.

## Installation

```sh
pip install .
```

## Test

```sh
python tests/test_bitlinear158_inference.py
```
