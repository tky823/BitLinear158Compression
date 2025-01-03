#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace bitlinear158compression
{
    template <typename scalar_t>
    __global__ void bitlinear158_inference_cuda_forward_kernel(const scalar_t *input, const int8_t *quantized_weight, scalar_t *output, const int64_t batch_size, const int64_t in_features, const int64_t out_features)
    {
        const int batch_idx = blockIdx.y;
        const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < batch_size && out_idx < out_features)
        {
            for (int64_t in_idx = 0; in_idx < in_features; in_idx++)
            {
                const scalar_t _input = input[in_features * batch_idx + in_idx];
                const int8_t _quantized_weight = quantized_weight[in_features * out_idx + in_idx];

                if (_quantized_weight > 0)
                {
                    output[out_features * batch_idx + out_idx] += _input;
                }
                else if (_quantized_weight < 0)
                {
                    output[out_features * batch_idx + out_idx] -= _input;
                }
            }
        }
    }

    at::Tensor bitlinear158_inference_cuda_forward(
        const at::Tensor &input, const at::Tensor &quantized_weight)
    {
        TORCH_CHECK(input.dim() > 1, "Input should NOT be scalar.");

        at::Tensor contiguous_input = input.contiguous().view({-1, input.size(-1)});

        TORCH_CHECK(quantized_weight.dim() == 2, "Quantized weight should be two dimensional.");
        TORCH_CHECK(quantized_weight.dtype() == at::kChar, "Quantized weight should be int8.");
        TORCH_CHECK(quantized_weight.size(1) == contiguous_input.size(1), "Numbers of channels are different among input and quanrized weight.");

        at::Tensor contiguous_quantized_weight = quantized_weight.contiguous();

        TORCH_INTERNAL_ASSERT(contiguous_input.device().type() == at::DeviceType::CUDA, "Input should be on CUDA.");
        TORCH_INTERNAL_ASSERT(contiguous_quantized_weight.device().type() == at::DeviceType::CUDA, "Quantized weight should be on CUDA.");

        const int64_t batch_size = contiguous_input.size(0);
        const int64_t in_features = contiguous_input.size(1);
        const int64_t out_features = quantized_weight.size(0);

        std::vector<int64_t> output_shape = std::vector<int64_t>(input.sizes().begin(), input.sizes().end() - 1);
        output_shape.push_back(out_features);
        at::Tensor output = torch::zeros(output_shape, input.options());

        const int threads = 1024;
        const dim3 blocks((out_features + threads - 1) / threads, batch_size);

        AT_DISPATCH_FLOATING_TYPES(
            input.type(),
            "bitlinear158_inference_cuda_forward",
            ([&]
             { bitlinear158_inference_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                   contiguous_input.data_ptr<scalar_t>(),
                   contiguous_quantized_weight.data_ptr<int8_t>(),
                   output.data_ptr<scalar_t>(),
                   batch_size,
                   in_features,
                   out_features); }));

        return output;
    }

    TORCH_LIBRARY_IMPL(bitlinear158compression, CUDA, m)
    {
        m.impl("bitlinear158_inference_forward", &bitlinear158compression::bitlinear158_inference_cuda_forward);
    }
}
