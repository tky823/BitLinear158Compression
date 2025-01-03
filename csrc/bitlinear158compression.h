#ifndef BITLINEAR158COMPRESSION_H
#define BITLINEAR158COMPRESSION_H

#include <torch/extension.h>

namespace bitlinear158compression
{
    std::vector<at::Tensor> bitlinear158_inference_cpu_forward(const at::Tensor &input, const at::Tensor &quantized_weight);
}

#endif // BITLINEAR158COMPRESSION_H
