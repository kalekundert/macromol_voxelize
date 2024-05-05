import polars as pl
import numpy as np

from ._voxelize import (
        Sphere, Atom, Grid,
        _add_atom_to_image, _get_voxel_center_coords,
)
from dataclasses import dataclass

from typing import TypeAlias, Callable
from numpy.typing import NDArray

"""\
Data structures and naming conventions
======================================
This list only includes data types that don't have their own classes.

`img`:
  A `np.ndarray` that contains the voxelized scene.

`atoms`:
  A `pandas.DataFrame` with the following columns: x, y, z, element.  This is 
  the way atoms are expected to be represented externally.  Internally, the 
  Atom data class is used to represent individual atoms.

`voxel`:
  A 3D `numpy.ndarray` containing the index for one of the cells in the image.  
  Generally, these indices are constrained to actually fall within the image in 
  question (i.e. no indices greater than the size of the image or less than 0).  
  Note that a `grid` object is needed to determine the physical location of the 
  voxel.  When multiple voxels are involved, the array has dimensions of (N,3).

`coords`:
  A 3D `numpy.ndarray` containing the location of the center of a voxel in 
  real-world coordinates, in units of Angstroms.  When multiple coordinates 
  that involved, the array has dimensions of (N,3).
"""

@dataclass
class ImageParams:
    """\
    A collection of parameters that apply to the image as a whole, as opposed
    to individual atoms.

    The most important parameters are `channels` and `grid`.  Together, these 
    specify the dimensions of the image.  The remaining parameters have 
    reasonable defaults.
    """

    channels: int
    """\
    The number of channels in the image.

    Note that this must be consistent with the *channels* column of the *atoms* 
    data frame passed to `image_from_atoms()`.  An error will be raised if any 
    atoms have channel indices that exceed the actual number of channels, or 
    are negative.
    """

    grid: Grid
    """\
    The spatial dimensions of the image.
    """

    dtype: type[np.floating] = np.float32
    """\
    The data type used to encode each voxel of the image.

    The following data types are supported:

    - `np.float32`, a.k.a. `np.single`
    - `np.float64`, a.k.a. `np.double`

    Note that 64-bit (i.e. double-precision) floating point numbers are always 
    used for the intermediate calculations needed to fill in each voxel.  
    According to the overlap_ library, which implements most of these 
    calculations, "reducing the numerical precision of the scalar floating 
    point type will have a significant impact on the precision and stability of 
    the calculations".  Therefore, this setting only affects the precision used 
    to store the final result, and in turn the size of the final image.
    """

    assign_channels: Callable[[pl.DataFrame], pl.DataFrame] = lambda x: x
    """\
    A function that can be used to assign channels the atoms being voxelized.

    This function is called after an initial filtering step that removes many 
    of the atoms that are too far away to be included in the image.  As such, 
    this can be slightly more efficient that assigning channels for every atom 
    in advance.  

    This function is passed the filtered *atoms* data frame and should return a 
    modified version of the same.  Do not modify any values in the *x*, *y*, 
    *z*, or *radius_A* columns, despite the possibility of doing so.  These 
    values have already been used in the aforementioned filtering steps, and 
    changing them here would lead to inconsistent results.
    """

Image: TypeAlias = NDArray

def image_from_atoms(atoms: pl.DataFrame, img_params: ImageParams) -> Image:
    """\
    Create an voxelized representation of the given atoms.

    Arguments:
        atoms:
            A dataframe representing the atoms to voxelize.  The following 
            columns are used to build the image.  Any other columns will be 
            silently ignored:

            - *x*, *y*, *z* (required): The center coordinates of each atom, in 
              units of angstroms.

            - *radius_A* (required): The radius of each atom, in units of 
              angstroms.  The `set_atom_radius_A()` function can be used to 
              create this column, if necessary.

            - *channels* (required): A list of integers specifying the channels 
              that each atom belongs to.  To be clear, each atom can belong to 
              any number of channels.  Each channel index must be between 0 and 
              ``img_params.channels - 1``.  Note that this column doesn't have 
              to be present in the *atoms* dataframe; it can also be calculated 
              by `img_params.assign_channels` after an initial filtering step.

            - *occupancy* (optional): How "present" each atom is.  More 
              specifically, this is a factor that will be used to scale the 
              overlap between the atom and each voxel.  If not specified, an 
              occupancy of 1 is assumed.

        img_params:
            An object specifying any information that applies to the image as a 
            whole, rather than to individual atoms.  This most importantly 
            includes the dimensions of the image.

    Returns:
        A floating point array of dimension $(C, X, Y, Z)$, where $C$ is 
        the number of channels specified by `img_params.channels` and $X$, $Y$, 
        and $Z$ are the spatial dimensions specified by 
        `img_params.grid.length_voxels`.
    """
    img = _make_empty_image(img_params)
    grid = img_params.grid

    # Without this filter, `_find_voxels_possibly_contacting_sphere()` becomes 
    # a performance bottleneck.
    atoms = _discard_atoms_outside_image(atoms, grid)
    atoms = img_params.assign_channels(atoms)

    _check_channels(atoms, img_params.channels)

    for row in atoms.iter_rows(named=True):
        atom = _make_atom(row)
        _add_atom_to_image(img, grid, atom)

    return img
        
def set_atom_radius_A(atoms: pl.DataFrame, radius_A: float):
    """\
    Assign all atoms the same radius.

    Arguments:
        atoms:
            A dataframe representing the atoms to voxelize.

        radius_A:
            The radius to assign, in angstroms.

    Returns:
        The input dataframe, with a new *radius_A* column.  Every row in this 
        column will have the same value.
    """
    return atoms.with_columns(radius_A=radius_A)

def set_atom_channels_by_element(
        atoms: pl.DataFrame,
        channels: list[str],
        first_match: bool = False,
        allow_missing_atoms: bool = False,
):
    """\
    Assign atoms to channels based on their element types.

    Arguments:
        atoms:
            A dataframe representing the atoms to voxelize.  This function 
            requires a column named "element", which must contain element names 
            as strings.

        channels:
            A list of regular expression patterns, each of which should match 
            elements that belong in a particular channel.  The *first_match* 
            argument controls what happens if multiple patterns match the same 
            atom.

            For example, ``['C', 'N', 'O', '.*']`` would place carbon in the 
            first channel, nitrogen in the second channel, and oxygen in the 
            third channel.  The fourth channel would contain all atoms 
            (including those already in one of the first three channels) if 
            *first_match=False*, otherwise it contain just those atoms that 
            aren't in of the earlier channels.

        first_match:
            If *True*, only allow each atom to occupy one channel: the first 
            one it matches.  If *False*, each atom will occupy every channel it 
            matches.

        allow_missing_atoms:
            If *True*, atoms that aren't assigned to any channel will be 
            silently removed.  By default, an error will be raised if any such 
            atoms are present.

    Returns:
        The input dataframe, with a *channels* column added.  Each entry in 
        this column will be a list of integers, where each integer identifies a 
        single channel and will be in the range [0, ``len(channels) - 1``].
    """
    channel_exprs = [
            pl.when(
                pl.col('element').str.contains(f'^({pattern})$')
            )
            .then(i)
            for i, pattern in enumerate(channels)
    ]
    if first_match:
        channel_exprs = pl.coalesce(channel_exprs)

    atoms = (
            atoms
            .with_columns(
                channels=pl.concat_list(channel_exprs).list.drop_nulls()
            )
    )

    if allow_missing_atoms:
        return atoms.filter(pl.col('channels').list.len() > 0)

    have_missing_atoms = (
            atoms
            .select(
                (pl.col('channels').list.len() == 0).any()
            )
            .item()
    )
    if have_missing_atoms:
        missing_elements = (
                atoms
                .filter(pl.col('channels').list.len() == 0)
                .get_column('element')
                .unique()
                .to_list()
        )
        raise ValidationError(f"""\
all atoms must be assigned at least one channel
• channels: {channels!r}
✖ unassigned elements: {missing_elements!r}
""")

    return atoms

def get_voxel_center_coords(grid, voxels):
    """\
    Calculate the center coordinates of the given voxels.

    Arguments:
        grid:
            An object specifying the size and location of each voxel.
            
        voxels:
            An integer array of dimension (N, 3) specifying the indices of 
            the voxels to calculate coordinates for.
    """
    # There are two things to keep in mind when passing arrays between 
    # python/numpy and C++/Eigen:
    #
    # - Coordinates and voxel indices are represented as row vectors by 
    #   python/numpy, and as column vectors by C++/Eigen.  This means that 
    #   arrays have to be transposed when moving from one language to the 
    #   other.  In principle, it would be possible to use the same row/column 
    #   vector convention in both languages.  But this would make it harder to 
    #   interact with third-party libraries like `overlap`.
    #
    # - Eigen doesn't have 1D arrays.  Instead it has vectors, which are just 
    #   2D matrices with either 1 row or 1 column.  When converting a vector 
    #   from C++/Eigen back to python/numpy, it's not clear whether the 
    #   resulting array should be 1D or 2D.  This ambiguity can be resolved by 
    #   looking at the shape of the original numpy input.
    #
    # I decided against accounting for either of these issues in the binding 
    # code itself.  The main reason for exposing most of the C++ functions to 
    # python is testing, and for that it's not helpful to be changing the 
    # inputs and outputs.  But this specific function is useful in other 
    # contexts, so I wrote this wrapper function to enforce the python 
    # conventions.

    coords_A = _get_voxel_center_coords(grid, voxels.T).T
    return coords_A.reshape(voxels.shape)


def _check_channels(atoms, num_channels):
    channels = atoms['channels'].explode()
    if not channels.is_empty():
        assert channels.min() >= 0
        assert channels.max() < num_channels

def _make_empty_image(img_params):
    shape = img_params.channels, *img_params.grid.shape
    return np.zeros(shape, dtype=img_params.dtype)

def _discard_atoms_outside_image(atoms, grid):
    max_r = atoms['radius_A'].max()

    min_corner = grid.center_A - (grid.length_A / 2 + max_r)
    max_corner = grid.center_A + (grid.length_A / 2 + max_r)

    return atoms.filter(
            pl.col('x') > min_corner[0],
            pl.col('x') < max_corner[0],
            pl.col('y') > min_corner[1],
            pl.col('y') < max_corner[1],
            pl.col('z') > min_corner[2],
            pl.col('z') < max_corner[2],
    )

def _make_atom(row):
    return Atom(
            sphere=Sphere(
                center_A=np.array([row['x'], row['y'], row['z']]),
                radius_A=row['radius_A'],
            ),
            channels=row['channels'],
            occupancy=row.get('occupancy', 1.0),
    )

class ValidationError(Exception):
    pass
