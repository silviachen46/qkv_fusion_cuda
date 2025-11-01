from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='qkv_fused',
    ext_modules=[
        CUDAExtension(
            name='qkv_fused',
            sources=[
                'my_ops/binding.cpp',
                'my_ops/qkv_fused_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3', '--use_fast_math',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_75,code=sm_75',
                ],
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
