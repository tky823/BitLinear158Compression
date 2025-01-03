#include "bitlinear158compression.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(bitlinear158compression, m)
{
    m.def("bitlinear158_inference_forward(Tensor input, Tensor quantized_weight) -> Tensor");
    m.def("bitlinear158_inference_backward(Tensor input, Tensor quantized_weight, Tensor grad_output) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(bitlinear158compression, CPU, m)
{
    m.impl("bitlinear158_inference_forward", &bitlinear158compression::bitlinear158_inference_cpu_forward);
    m.impl("bitlinear158_inference_backward", &bitlinear158compression::bitlinear158_inference_cpu_backward);
}
