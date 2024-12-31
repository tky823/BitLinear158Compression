# BitLinear158Compression

In this library, we compare following compression models for inference by BitLinear158:

- `float32`:Ternary weight is managed by `float32` (naive implementation)
- `int8`: Ternary weight is managed by `int8`
- `uint2`: Ternary weight is managed by `uint2` defined in `torchao` 

## Installation

```sh
pip install .
```

## Test

```sh
python tests/test_bitlinear158_inference.py
```
