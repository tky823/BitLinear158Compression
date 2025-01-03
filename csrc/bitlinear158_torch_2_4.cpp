/*
    for C++ extension with torch >= 2.4
*/
#include "bitlinear158compression.h"

namespace bitlinear158compression
{
    std::vector<at::Tensor> bitlinear158_inference_cpu_forward(const at::Tensor &input, const at::Tensor &quantized_weight)
    {
        at::Tensor output = torch::linear(input, quantized_weight);

        return {output};
    }
}
