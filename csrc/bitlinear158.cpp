#include <torch/extension.h>
#include <vector>

#ifdef WITH_CUDA
std::vector<torch::Tensor> bitlinear158_inference_cuda_forward(
    torch::Tensor input,
    torch::Tensor quantized_weight);
std::vector<torch::Tensor> bitlinear158_inference_cuda_backward(
    torch::Tensor input,
    torch::Tensor quantized_weight,
    torch::Tensor grad_output);
#endif

std::vector<torch::Tensor> bitlinear158_inference_cpu_forward(
    torch::Tensor input,
    torch::Tensor quantized_weight)
{
    torch::Tensor output = torch::linear(input, quantized_weight);

    return {output};
}

std::vector<torch::Tensor> bitlinear158_inference_cpu_backward(
    torch::Tensor input,
    torch::Tensor quantized_weight,
    torch::Tensor grad_output)
{
    torch::Tensor grad_input = torch::matmul(grad_output, quantized_weight);
    torch::Tensor grad_weight = torch::matmul(grad_output.transpose(1, 0), input);

    return {grad_input, grad_weight};
}

std::vector<torch::Tensor> bitlinear158_inference_forward(
    torch::Tensor input,
    torch::Tensor quantized_weight)
{
    if (input.is_cuda())
    {
#ifdef WITH_CUDA
        return bitlinear158_inference_cuda_forward(input, quantized_weight);
#endif
        AT_ERROR("CUDA is not available, but given tensor is on CUDA.");
    }
    else
    {
        return bitlinear158_inference_cpu_forward(input, quantized_weight);
    }
}

std::vector<torch::Tensor> bitlinear158_inference_backward(
    torch::Tensor input,
    torch::Tensor quantized_weight,
    torch::Tensor grad_output)
{
    if (input.is_cuda())
    {
#ifdef WITH_CUDA
        return bitlinear158_inference_cuda_backward(input, quantized_weight, grad_output);

#endif
        AT_ERROR("CUDA is not available, but given tensor is on CUDA.");
    }
    else
    {
        return bitlinear158_inference_cpu_backward(input, quantized_weight, grad_output);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &bitlinear158_inference_forward, "bitlinear158 inference forward");
    m.def("backward", &bitlinear158_inference_backward, "bitlinear158 inference backward");
}