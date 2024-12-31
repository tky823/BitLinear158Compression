#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void bitlinear158_inference_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input,
    const torch::PackedTensorAccessor<int8_t, 2, torch::RestrictPtrTraits, size_t> quantized_weight,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output)
{
    const int batch_idx = blockIdx.y;
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_size = input.size(0);
    const int in_features = quantized_weight.size(1);
    const int out_features = quantized_weight.size(0);

    if (batch_idx < batch_size && out_idx < out_features)
    {
        for (auto in_idx = 0; in_idx < in_features; in_idx++)
        {
            if (quantized_weight[out_idx][in_idx] > 0)
            {
                output[batch_idx][out_idx] += input[batch_idx][in_idx];
            }
            else if (quantized_weight[out_idx][in_idx] < 0)
            {
                output[batch_idx][out_idx] -= input[batch_idx][in_idx];
            }
        }
    }
}

template <typename scalar_t>
__global__ void bitlinear158_inference_input_backward_kernel(
    const torch::PackedTensorAccessor<int8_t, 2, torch::RestrictPtrTraits, size_t> quantized_weight,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> grad_output,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> grad_input)
{
    const int batch_idx = blockIdx.y;
    const int in_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int in_features = quantized_weight.size(1);
    const int out_features = quantized_weight.size(0);

    if (in_idx < in_features)
    {
        for (auto out_idx = 0; out_idx < out_features; out_idx++)
        {
            grad_input[batch_idx][in_idx] += grad_output[batch_idx][out_idx] * static_cast<scalar_t>(quantized_weight[out_idx][in_idx]);
        }
    }
}

template <typename scalar_t>
__global__ void bitlinear158_inference_quantized_weight_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> grad_output,
    torch::PackedTensorAccessor<int8_t, 2, torch::RestrictPtrTraits, size_t> grad_quantized_weight)
{
    const int out_idx = blockIdx.y;
    const int in_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_size = input.size(0);
    const int in_features = input.size(1);

    if (in_idx < in_features)
    {
        for (auto batch_idx = 0; batch_idx < batch_size; batch_idx++)
        {
            grad_quantized_weight[out_idx][in_idx] += grad_output[batch_idx][out_idx] * input[batch_idx][in_idx];
        }
    }
}

std::vector<torch::Tensor> bitlinear158_inference_cuda_forward(
    torch::Tensor input,
    torch::Tensor quantized_weight)
{
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = quantized_weight.size(0);
    torch::Tensor output = torch::zeros({batch_size, out_features}, input.options());

    const int threads = 1024;
    const dim3 blocks((out_features + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(
        input.type(),
        "bitlinear158_inference_cuda_forward",
        ([&]
         { bitlinear158_inference_forward_kernel<scalar_t><<<blocks, threads>>>(
               input.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
               quantized_weight.packed_accessor<int8_t, 2, torch::RestrictPtrTraits, size_t>(),
               output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()); }));

    return {output};
}

std::vector<torch::Tensor> bitlinear158_inference_cuda_backward(
    torch::Tensor input,
    torch::Tensor quantized_weight,
    torch::Tensor grad_output)
{
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = quantized_weight.size(0);
    torch::Tensor grad_input = torch::zeros_like(input);
    torch::Tensor grad_quantized_weight = torch::zeros_like(quantized_weight);

    const int threads = 1024;
    const dim3 input_blocks((in_features + threads - 1) / threads, batch_size);
    const dim3 weight_blocks((in_features + threads - 1) / threads, out_features);

    AT_DISPATCH_FLOATING_TYPES(
        input.type(),
        "bitlinear158_inference_cuda_input_backward",
        ([&]
         { bitlinear158_inference_input_backward_kernel<scalar_t><<<input_blocks, threads>>>(
               quantized_weight.packed_accessor<int8_t, 2, torch::RestrictPtrTraits, size_t>(),
               grad_output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
               grad_input.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()); }));

    AT_DISPATCH_FLOATING_TYPES(
        input.type(),
        "bitlinear158_inference_cuda_quantized_weight_backward",
        ([&]
         { bitlinear158_inference_quantized_weight_backward_kernel<scalar_t><<<weight_blocks, threads>>>(
               input.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
               grad_output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
               grad_quantized_weight.packed_accessor<int8_t, 2, torch::RestrictPtrTraits, size_t>()); }));

    return {grad_input, grad_quantized_weight};
}
