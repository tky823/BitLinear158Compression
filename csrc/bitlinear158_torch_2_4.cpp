/*
    for C++ extension with torch >= 2.4
*/
#include "bitlinear158compression.h"

namespace bitlinear158compression
{
    at::Tensor bitlinear158_inference_cpu_forward(const at::Tensor &input, const at::Tensor &quantized_weight)
    {
        at::Tensor output = torch::linear(input, quantized_weight);

        return output;
    }

    std::vector<torch::Tensor> bitlinear158_inference_cpu_backward(const at::Tensor &input, const at::Tensor &quantized_weight, const at::Tensor &grad_output)
    {
        at::Tensor grad_input = torch::matmul(grad_output, quantized_weight);
        at::Tensor grad_weight = torch::matmul(grad_output.transpose(1, 0), input);

        return {grad_input, grad_weight};
    }
}