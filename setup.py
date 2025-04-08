from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch

setup(
    name='pointpillars',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name='pointpillars.ops.voxel_op',
            sources=[
                'pointpillars/ops/voxelization/voxelization.cpp',
                'pointpillars/ops/voxelization/voxelization_cpu.cpp',
            ],
            include_dirs=[torch.utils.cpp_extension.include_paths()[0]]  # ✅ Added this
        ),
        CppExtension(
            name='pointpillars.ops.iou3d_op',
            sources=[
                'pointpillars/ops/iou3d/iou3d.cpp',
            ],
            include_dirs=[torch.utils.cpp_extension.include_paths()[0]]  # ✅ Added this
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False
)