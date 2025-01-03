import time

import torch

from bitlinear158compression.bitnet import (
    BitLinear158,
    BitLinear158CppTraining,
)


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
    cpp_training_model = BitLinear158CppTraining.build_from_bitlinear158(training_model)

    training_model.to(device)
    cpp_training_model.to(device)

    training_elapsed_time = 0
    cpp_training_elapsed_time = 0

    for iter_idx in range(num_iter):
        input = torch.randn((batch_size, in_features))
        target = torch.randn((batch_size, out_features))
        input = input.to(device)
        target = target.to(device)

        start = time.perf_counter()
        training_output = training_model(input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        training_elapsed_time += end - start

        start = time.perf_counter()
        cpp_training_output = cpp_training_model(input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        cpp_training_elapsed_time += end - start

        if iter_idx == 0:
            assert torch.allclose(training_output, cpp_training_output, atol=1e-7)

            training_loss = torch.mean(target - training_output)
            cpp_training_loss = torch.mean(target - cpp_training_output)

            training_loss.backward()
            cpp_training_loss.backward()

            assert torch.allclose(
                training_model.weight, cpp_training_model.weight, atol=1e-7
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

    print("Total training time:")
    print(f"\tfloat32: {training_elapsed_time}")
    print(f"\t    cpp: {cpp_training_elapsed_time}")


if __name__ == "__main__":
    main()
