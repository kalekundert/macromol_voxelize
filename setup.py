from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
import pyarrow as pa
import sys

pa.create_library_symlinks()

if sys.platform == "darwin":
    extra_link_args = ["-Wl,-rpath,@loader_path/../pyarrow"]
else:
    extra_link_args = ["-Wl,-rpath,$ORIGIN/../pyarrow"]

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
                    pa.get_include(),
                ],
                libraries=pa.get_libraries(),
                library_dirs=pa.get_library_dirs(),
                extra_link_args=extra_link_args,
                cxx_std=17,
            ),
        ],
)
