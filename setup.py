from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

setup(
        ext_modules=[
            Pybind11Extension(
                name='macromol_voxelize._voxelize',
                sources=[
                    'src/_voxelize.cc',
                ],
                include_dirs=[
                    'src/vendored/Eigen',
                    'src/vendored/overlap',
                ],
                cxx_std=14,
            ),
        ],
)
