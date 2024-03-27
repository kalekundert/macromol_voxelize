from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

setup(
        ext_modules=[
            Pybind11Extension(
                name='macromol_voxelize._voxelize',
                sources=[
                    'macromol_voxelize/_voxelize.cc',
                ],
                include_dirs=[
                        'macromol_voxelize/vendored/Eigen',
                        'macromol_voxelize/vendored/overlap',
                ],
                cxx_std=14,
            ),
        ],
)
