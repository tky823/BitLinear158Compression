import torch

from bitlinear158compression.bitnet import (
    BitLinear158,
    BitLinear158Int8Inference,
    BitLinear158Uint2Inference,
)
from bitlinear158compression.utils import compute_model_size, compute_state_dict_size


def main() -> None:
    torch.manual_seed(0)

    batch_size = 4
    in_features = 256
    out_features = 512

    input = torch.randn((batch_size, in_features))

    bias = True
    weight_only_quantization = True
    training_model = BitLinear158(
        in_features,
        out_features,
        bias=bias,
        weight_only_quantization=weight_only_quantization,
    )
    int8_inference_model = BitLinear158Int8Inference.build_from_bitlinear158(
        training_model
    )
    uint2_inference_model = BitLinear158Uint2Inference.build_from_bitlinear158(
        training_model
    )

    training_model.eval()
    int8_inference_model.eval()
    uint2_inference_model.eval()

    with torch.inference_mode():
        training_output = training_model(input)
        int8_inference_output = int8_inference_model(input)
        uint2_inference_output = uint2_inference_model(input)

    assert torch.allclose(training_output, int8_inference_output)
    assert torch.allclose(training_output, uint2_inference_output)

    print("Size of model parameters in byte:")
    print("\tfloat32: {:8d}".format(compute_model_size(training_model)))
    print("\t   int8: {:8d}".format(compute_model_size(int8_inference_model)))
    print("\t  uint2: {:8d}".format(compute_model_size(uint2_inference_model)))

    print("Size of stored state_dict in byte:")
    print("\tfloat32: {:8d}".format(compute_state_dict_size(training_model)))
    print("\t   int8: {:8d}".format(compute_state_dict_size(int8_inference_model)))
    print("\t  uint2: {:8d}".format(compute_state_dict_size(uint2_inference_model)))


if __name__ == "__main__":
    main()
