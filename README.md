Macromolecular Voxelization
===========================

[![Last release](https://img.shields.io/pypi/v/macromol_voxelize.svg)](https://pypi.python.org/pypi/macromol_voxelize)
[![Python version](https://img.shields.io/pypi/pyversions/macromol_voxelize.svg)](https://pypi.python.org/pypi/macromol_voxelize)
[![Documentation](https://img.shields.io/readthedocs/macromol_voxelize.svg)](https://macromol-voxelize.readthedocs.io/en/latest)
[![Test status](https://img.shields.io/github/actions/workflow/status/kalekundert/macromol_voxelize/test.yml?branch=master)](https://github.com/kalekundert/macromol_voxelize/actions)
[![Test coverage](https://img.shields.io/codecov/c/github/kalekundert/macromol_voxelize)](https://app.codecov.io/github/kalekundert/macromol_voxelize)
[![Last commit](https://img.shields.io/github/last-commit/kalekundert/macromol_voxelize?logo=github)](https://github.com/kalekundert/macromol_voxelize)

*Macromol Voxelize* is a highly performant library for converting atomic 
structures into 3D images, i.e. images where each channel might represent a 
different element type, and each voxel might be on the order of 1Å in each 
dimension.  The intended use case is machine learning.  More specifically, it 
is to allow image-based model architectures (such as CNNs) to be applied to 
macromolecular data.

Some noteworthy aspects of this library:

- Algorithm: The voxelization procedure implemented by this library is to (i) 
  treat each atom as a sphere and (ii) fill each voxel in proportion to amount 
  it overlaps that sphere.  Although this procedure may seem intuitive, it's 
  actually quite unique.  Macromolecular structures are typically voxelized in 
  one of two ways: either by assigning the entire density for each atom to a 
  single voxel, or by modeling each atom as a 3D Gaussian distribution.  The 
  advantage of the overlap-based procedure is that the image changes more 
  smoothly as atoms move around.  It also makes it easier to infer the exact 
  position of each atom from just the image.  The disadvantage is that 
  calculating sphere/cube overlap volumes turns out to be quite difficult. 
  Here, the [`overlap`](https://github.com/severinstrobl/overlap) library is 
  used to make this calculation.

- Performance: Because voxelization can be a bottleneck during training, most 
  of this library is implemented in C++.  However, the API is in Python, for 
  compatibility with common machine learning frameworks such as PyTorch, JAX, 
  etc.  Note that the voxelization algorithm is deliberately single-threaded. 
  This is a bit counter-intuitive, since voxelization is an embarrassingly 
  parallel problem.  However, in the context of loading training examples, it's 
  more efficient to have a larger number of single-threaded data loader 
  subprocesses than a smaller number of multi-threaded ones.

Here's an example showing how to voxelize a set of atoms:

```python
import polars as pl
import macromol_voxelize as mmvox

# Load the atoms in question.  These particular coordinates are for a 
# methionine amino acid.  This is a hard-coded dataframe for simplicity, but 
# normally you'd use <https://github.com/kalekundert/macromol_dataframe> for 
# this step.
atoms = pl.DataFrame([
        dict(element='N', x= 1.052, y=-1.937, z=-1.165),
        dict(element='C', x= 1.540, y=-0.561, z=-1.165),
        dict(element='C', x= 3.049, y=-0.521, z=-1.165),
        dict(element='O', x= 3.733, y=-1.556, z=-1.165),
        dict(element='C', x= 0.965, y= 0.201, z= 0.059),
        dict(element='C', x=-0.570, y= 0.351, z= 0.100),
        dict(element='S', x=-1.037, y= 1.495, z= 1.409),
        dict(element='C', x=-2.800, y= 1.146, z= 1.451),
])

# Add a "radius_A" column to the dataframe.  (The "_A" suffix means "in units 
# of angstroms".)  This function simply gives each atom the same radius, but 
# you could calculate radii however you want.
atoms = mmvox.set_atom_radius_A(atoms, 0.75)

# Add a "channels" column to the dataframe.  This function assigns channels by 
# matching each atom's element name to an index in the given list, but again 
# you could do this however you want.
atoms = mmvox.set_atom_channels_by_element(atoms, [['C'], ['N'], ['O'], ['S', 'SE']])

# Create the 3D image.  Note that this step is not specific to macromolecules 
# in any way.  It just expects a data frame with "x", "y", "z", "radius_A", 
# "occupancy", and "channels" columns.
img_params = mmvox.ImageParams(
        channels=4,
        grid=mmvox.Grid(
            length_voxels=8,
            resolution_A=1,
            center_A=[0, 0, 0],
        ),
)
img, img_atoms = mmvox.image_from_atoms(atoms, img_params)
```

This library also contains tools for rendering these images in PyMOL.  Here's 
what the above image looks like:

<p align="center">
  <img src="docs/met.png" width="400">
</p>

Refer to the [online 
documentation](https://macromol-voxelize.readthedocs.io/en/latest) for more 
information.
