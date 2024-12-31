import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension as _BuildExtension
from torch.utils.cpp_extension import CppExtension, CUDAExtension


class BuildExtension(_BuildExtension):
    # TODO: support editable mode
    extensions = []

    if torch.cuda.is_available():
        extensions.append(
            CUDAExtension(
                name="bitlinear158compression._C.bitlinear158",
                sources=[
                    "csrc/bitlinear158.cpp",
                    "csrc/cuda/bitlinear158_cuda.cu",
                ],
            )
        )
    else:
        extensions.append(
            CppExtension(
                name="bitlinear158compression._C.bitlinear158",
                sources=[
                    "csrc/bitlinear158.cpp",
                ],
            )
        )


setup(
    ext_modules=BuildExtension.extensions,
    cmdclass={"build_ext": BuildExtension},
)
