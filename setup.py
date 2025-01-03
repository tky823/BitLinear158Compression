import os

import torch
from packaging import version
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension as _BuildExtension
from torch.utils.cpp_extension import CppExtension, CUDAExtension


class BuildExtension(_BuildExtension):
    extensions: list[CppExtension | CUDAExtension] = []

    if torch.cuda.is_available():
        extensions.append(
            CUDAExtension(
                name="bitlinear158compression._C.bitlinear158",
                sources=[
                    "csrc/bitlinear158.cpp",
                    "csrc/cuda/bitlinear158_cuda.cu",
                ],
                extra_compile_args=[
                    "-DWITH_CUDA",
                ],
            )
        )
    else:
        if version.parse(torch.__version__) >= version.parse("2.4.0"):
            extensions.append(
                CppExtension(
                    name="bitlinear158compression._C.bitlinear158",
                    sources=[
                        "csrc/main_torch_2_4.cpp",
                        "csrc/bitlinear158_torch_2_4.cpp",
                    ],
                ),
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

    def run(self) -> None:
        if self.editable_mode:
            # create directories to save ".so" files in editable mode.
            for extension in self.extensions:
                *pkg_names, _ = extension.name.split(".")
                os.makedirs("/".join(pkg_names), exist_ok=True)

        super().run()


setup(
    ext_modules=BuildExtension.extensions,
    cmdclass={"build_ext": BuildExtension},
)
