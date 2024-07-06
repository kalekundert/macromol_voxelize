import polars as pl
import numpy as np

from ._voxelize import Grid, _add_atoms_to_image, _get_voxel_center_coords
from dataclasses import dataclass

from typing import TypeAlias, Callable, Optional
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

    process_filtered_atoms: Callable[[pl.DataFrame], pl.DataFrame] = lambda x: x
    """\
    Calculate any remaining parameters needed to construct the image, but only 
    for those atoms that will actually be in the image.

    The main reason to use this function is to more efficiently assign 
    channels, and sometimes radii, to each atom.  It would be possible to 
    assign both these properties before calling `image_from_atoms()`, but that 
    would mean processing the whole *atoms* dataframe.  Most of the time, only 
    a small fraction of those atoms will end up in the final image, so 
    processing all of them is a waste of time.  In contrast, this function is 
    called after an initial filtering step removes irrelevant atoms.

    This function should never modify any values that were used in the 
    filtering step itself.  Doing so would likely cause the results of the 
    filtering step to be incorrect.  This always includes the *x*, *y*, and *z* 
    columns.  It also includes the *radius_A* column if `max_radius_A` isn't 
    specified.
    """

    max_radius_A: Optional[float] = None
    """\
    The maximum radius to use when filtering atoms that are outside the image, 
    in units of angstroms.

    If not specified, the maximum radius will be calculated from the *radius_A* 
    column of the *atoms* dataframe passed to `image_from_atoms()`.  The main 
    reason to specify this parameter is to allow *radius_A* to be calculated by 
    `process_filtered_atoms()`, which can be more efficient.  Note that an error 
    will be raised if any atoms in the image have radii larger than this 
    maximum.
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
              that each atom belongs to.  Each atom can belong to any number of 
              channels, and each channel index must be between 0 and 
              ``img_params.channels - 1``.  Note that this column doesn't have 
              to be present in the *atoms* dataframe; it can also be calculated 
              by `img_params.process_filtered_atoms()`.

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
    atoms = _discard_atoms_outside_image(atoms, grid, img_params.max_radius_A)
    atoms = img_params.process_filtered_atoms(atoms)

    if __debug__:
        _check_channels(atoms, img_params.channels)
        _check_max_radius_A(atoms, img_params.max_radius_A)

    if 'occupancy' not in atoms:
        atoms = atoms.with_columns(occupancy=1.0)

    # If the input dataframe is in the right format, the casts will be no-ops 
    # and the numpy conversions won't perform any copies.  However, this isn't 
    # generally the case.  Data types can depend on the library used to load 
    # the atoms, and filtering operations can prevent no-copy numpy conversions 
    # unless the dataframe is rechunked afterwards.
    _add_atoms_to_image(
            img,
            grid,
            atoms['x'].cast(pl.Float64).to_numpy(),
            atoms['y'].cast(pl.Float64).to_numpy(),
            atoms['z'].cast(pl.Float64).to_numpy(),
            atoms['radius_A'].cast(pl.Float64).to_numpy(),
            atoms['channels'].list.explode().cast(pl.Int64).to_numpy(),
            atoms['channels'].list.len().to_numpy(),  # no need for cast; len() always returns pl.Int32
            atoms['occupancy'].cast(pl.Float64).to_numpy(),
    )

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
    # Include the `float()` call to raise an error if an incompatible type is 
    # provided, instead of silently filling the dataframe with nonsense.
    return atoms.with_columns(radius_A=float(radius_A))

def set_atom_channels_by_element(
        atoms: pl.DataFrame,
        channels: list[list[str]],
        *,
        drop_missing_atoms: bool = False,
) -> pl.DataFrame:
    """\
    Assign atoms to channels based on their element types.

    Arguments:
        atoms:
            A dataframe representing the atoms to voxelize.  This function 
            requires a column named "element", which must contain element names 
            as strings.

        channels:
            A list of lists of element names.  Each item in the outer list 
            represents a different channel.  Each item in one of the inner 
            lists represents an element that should appear in said channel.  
            Each element can appear in any number of channels.  Furthermore, 
            the special symbol '*' can be used to represent any element that is 
            not mentioned explicitly.

            For example, consider: ``[['C'], ['N'], ['O'], ['S', 'SE']]``.  
            This indicates that carbon should go in the first channel, nitrogen 
            in the second, oxygen in the third, and both sulfur and selenium in 
            the fourth.  (Sulfur is commonly replaced by selenium in crystal 
            structures, to help solve the phasing problem.)

        drop_missing_atoms:
            If *True*, atoms that aren't assigned to any channel will be 
            silently removed.  By default, an error will be raised if any such 
            atoms are present.

    Returns:
        The input dataframe, with a *channels* column added.  Each entry in 
        this column will be a list of integers, where each integer identifies a 
        single channel and will be in the range [0, ``len(channels) - 1``].
    """

    channel_map = {}
    for i, elems in enumerate(channels):
        if isinstance(elems, str):
            raise ValidationError(f"expected list of elements, found str: {elems!r}")
        for elem in elems:
            channel_map.setdefault(elem, []).append(i)

    atoms = atoms.with_columns(
            channels=(
                pl.col('element')
                .replace_strict(
                    channel_map,
                    default=channel_map.pop('*', None),
                    return_dtype=pl.List(pl.Int64),
                )
            ),
    )

    if drop_missing_atoms:
        return atoms.drop_nulls('channels')

    elif atoms['channels'].null_count():
        missing_elements = (
                atoms
                .filter(pl.col('channels').is_null())
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

def add_atom_channel_by_expr(
        atoms: pl.DataFrame,
        expr: pl._typing.IntoExprColumn,
        channel: int,
):
    expr_channel = (
            pl.when(expr)
            .then([channel])
            .otherwise([])
    )
    return atoms.with_columns(
            channels=pl.col('channels').list.concat(expr_channel)
    )

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
        if (n := channels.min()) < 0 or (n := channels.max()) >= num_channels:
            raise ValidationError(f"channel indices must be between 0 and {num_channels - 1}, not {n}")

def _check_max_radius_A(atoms, max_radius_A):
    if max_radius_A is not None:
        if (atoms['radius_A'] > max_radius_A).any():
            raise ValidationError("atom radii must not exceed `ImageParams.max_radius_A`")

def _make_empty_image(img_params):
    shape = img_params.channels, *img_params.grid.shape
    return np.zeros(shape, dtype=img_params.dtype)

def _discard_atoms_outside_image(atoms, grid, max_radius_A):
    if max_radius_A is None:
        max_radius_A = atoms['radius_A'].max()

    min_corner = grid.center_A - (grid.length_A / 2 + max_radius_A)
    max_corner = grid.center_A + (grid.length_A / 2 + max_radius_A)

    return atoms.filter(
            pl.col('x') > min_corner[0],
            pl.col('x') < max_corner[0],
            pl.col('y') > min_corner[1],
            pl.col('y') < max_corner[1],
            pl.col('z') > min_corner[2],
            pl.col('z') < max_corner[2],
    )

class ValidationError(Exception):
    pass
