import time

import torch

from bitlinear158compression.bitnet import (
    BitLinear158,
    BitLinear158CppInference,
    BitLinear158Int8Inference,
    BitLinear158Uint2Inference,
)
from bitlinear158compression.utils import compute_model_size, compute_state_dict_size


def main() -> None:
    torch.manual_seed(0)

    num_iter = 10000
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    batch_size = 4
    in_features = 256
    out_features = 512

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
    cpp_inference_model = BitLinear158CppInference.build_from_bitlinear158(
        training_model
    )

    training_model.to(device)
    int8_inference_model.to(device)
    uint2_inference_model.to(device)
    cpp_inference_model.to(device)

    training_model.eval()
    int8_inference_model.eval()
    uint2_inference_model.eval()
    cpp_inference_model.eval()

    training_elapsed_time = 0
    int8_inference_elapsed_time = 0
    uint2_inference_elapsed_time = 0
    cpp_inference_elapsed_time = 0

    for iter_idx in range(num_iter):
        input = torch.randn((batch_size, in_features))
        input = input.to(device)

        with torch.inference_mode():
            start = time.perf_counter()
            training_output = training_model(input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()
            training_elapsed_time += end - start

            start = time.perf_counter()
            int8_inference_output = int8_inference_model(input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()
            int8_inference_elapsed_time += end - start

            start = time.perf_counter()
            uint2_inference_output = uint2_inference_model(input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()
            uint2_inference_elapsed_time += end - start

            start = time.perf_counter()
            cpp_inference_output = cpp_inference_model(input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()
            cpp_inference_elapsed_time += end - start

        if iter_idx == 0:
            assert torch.allclose(training_output, int8_inference_output)
            assert torch.allclose(training_output, uint2_inference_output)
            assert torch.allclose(training_output, cpp_inference_output, atol=1e-7)

            print("Size of model parameters in byte:")
            print(f"\tfloat32: {compute_model_size(training_model):8d}")
            print(f"\t   int8: {compute_model_size(int8_inference_model):8d}")
            print(f"\t  uint2: {compute_model_size(uint2_inference_model):8d}")
            print(f"\t    cpp: {compute_model_size(cpp_inference_model):8d}")

            print("Size of stored state_dict in byte:")
            print(f"\tfloat32: {compute_state_dict_size(training_model):8d}")
            print(f"\t   int8: {compute_state_dict_size(int8_inference_model):8d}")
            print(f"\t  uint2: {compute_state_dict_size(uint2_inference_model):8d}")
            print(f"\t    cpp: {compute_state_dict_size(cpp_inference_model):8d}")

    print("Total inference time:")
    print(f"\tfloat32: {training_elapsed_time}")
    print(f"\t   int8: {int8_inference_elapsed_time}")
    print(f"\t  uint2: {uint2_inference_elapsed_time}")
    print(f"\t    cpp: {cpp_inference_elapsed_time}")


if __name__ == "__main__":
    main()
