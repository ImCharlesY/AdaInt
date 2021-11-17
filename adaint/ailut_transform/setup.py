import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


csrc_directory = osp.join(osp.dirname(__file__), 'csrc')
setup(
    name='ailut',
    version='0.9',
    description='Adaptive Interval 3D LookUp Table Transform',
    ext_modules=[
        CUDAExtension('ailut', [
            osp.join(csrc_directory, 'ailut_transform_cuda.cpp'),
            osp.join(csrc_directory, 'ailut_transform_kernel.cu')
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
