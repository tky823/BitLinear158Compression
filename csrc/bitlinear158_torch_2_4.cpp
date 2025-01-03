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
        at::Tensor reshaped_input = input.view({-1, input.size(-1)});
        at::Tensor reshaped_grad_output = grad_output.view({-1, grad_output.size(-1)});
        at::Tensor grad_weight = torch::matmul(reshaped_grad_output.transpose(1, 0), reshaped_input);

        return {grad_input, grad_weight};
    }
}
