from __future__ import annotations

import polars as pl
import numpy as np

from ._voxelize import (
        Grid, FillAlgorithm, AggAlgorithm, _add_atoms_to_image,
        _get_voxel_center_coords, _find_voxels_containing_coords,
)
from dataclasses import dataclass

from typing import Optional, Type, Tuple, List
from typing_extensions import TypeAlias

try:
    from numpy.typing import NDArray
except ImportError:
    NDArray = np.ndarray

@dataclass
class ImageParams:
    """\
    A collection of parameters that apply to the image as a whole, as opposed
    to individual atoms.

    The most important parameters are `channels` and `grid`.  Together, these 
    specify the dimensions of the image.  The remaining parameters have 
    reasonable defaults.

    .. attribute:: channels
        :type: int

        The number of channels in the image.

        Note that this must be consistent with the *channels* column of the 
        *atoms* data frame passed to `image_from_atoms()`.  An error will be 
        raised if any atoms have channel indices that exceed the actual number 
        of channels, or are negative.

    .. attribute:: grid
        :type: Grid
        
        The spatial dimensions of the image.

    .. attribute:: dtype
        :type: Type[numpy.floating]
        :value: numpy.float32

        The data type used to encode each voxel of the image.

        The following data types are supported:

        - `numpy.float32`, a.k.a. `numpy.single`
        - `numpy.float64`, a.k.a. `numpy.double`

        Note that 64-bit (i.e. double-precision) floating point numbers are 
        always used for the intermediate calculations needed to fill in each 
        voxel.  According to the overlap_ library, which implements most of 
        these calculations, "reducing the numerical precision of the scalar 
        floating point type will have a significant impact on the precision and 
        stability of the calculations".  Therefore, this setting only affects 
        the precision used to store the final result, and in turn the size of 
        the final image.

    .. attribute:: max_radius_A
        :type: Optional[float]
        :value: None

        The maximum radius to use when filtering atoms that are outside the 
        image, in units of angstroms.

        If not specified, the maximum radius will be calculated from the 
        *radius_A* column of the *atoms* dataframe passed to 
        `image_from_atoms()`.  The main reason to specify this parameter is to 
        allow *radius_A* to be calculated after the initial filtering step, 
        which can be more efficient.  Note that an error will be raised if any 
        atoms in the image have radii larger than this maximum (and if 
        ``__debug__ == True``).

    .. attribute:: fill_algorithm
        :type: FillAlgorithm
        :value: FillAlgorithm.FractionAtom

        The algorithm used to calculate a value for each voxel in the image.  
        The following values are supported:

        - ``FillAlgorithm.OverlapA3``: The volume of overlap between the voxel 
          and any atoms, in units of Å³.

        - ``FillAlgorithm.FractionAtom``: The number (usually fractional) of 
          atoms contained in the voxel.  A nice property of this algorithm is 
          that the sum of all the voxels in the image will match the number of 
          atoms used to make it.

        - ``FillAlgorithm.FractionVoxel``: The fraction of the voxel that is 
          occupied by atoms.  This is calculated separately for each atom, 
          then summed, so the result can be greater than 1.

    .. attribute:: agg_algorithm
        :type: AggAlgorithm
        :value: AggAlgorithm.Sum

        The algorithm used to aggregate a new fill value for a voxel with the 
        existing value of that voxel.

        - ``AggAlgorithm.Sum``: Sum the two values.  This results in 
          meaningful volumes/fractions when a voxel is partially occupied by 
          two different atoms.

        - ``AggAlgorithm.Max``: Keep the higher of the two values.  This may 
          give more reasonable results than summing when there are lots of 
          overlapping atoms.

    .. _overlap: https://github.com/severinstrobl/overlap
    """

    channels: int
    grid: Grid
    dtype: Type[np.floating] = np.float32
    max_radius_A: Optional[float] = None
    fill_algorithm: FillAlgorithm = FillAlgorithm.FractionAtom
    agg_algorithm: AggAlgorithm = AggAlgorithm.Sum

Image: TypeAlias = NDArray

def image_from_atoms(atoms: pl.DataFrame, img_params: ImageParams) -> Tuple[Image, pl.DataFrame]:
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

            - *channels* (required): The channels that each atoms belongs to, 
              expressed as a list of integers.  Each atom can belong to any 
              number of channels.  Each channel index must be between 0 and 
              ``img_params.channels - 1``, inclusive.

            - *occupancy* (optional): How "present" each atom is.  More 
              specifically, this is a factor that will be used to scale the 
              overlap between the atom and each voxel.  If not specified, an 
              occupancy of 1 is assumed.

        img_params:
            An object specifying any information that applies to the image as a 
            whole, rather than to individual atoms.  This most importantly 
            includes the dimensions of the image.

    Returns:
        tuple:

        - image:
            A floating point array of dimension $(C, X, Y, Z)$, where $C$ is 
            the number of channels specified by ``img_params.channels`` and $X$, $Y$, 
            and $Z$ are the spatial dimensions specified by 
            ``img_params.grid.length_voxels``.

        - atoms:
            A dataframe containing just the atoms that are present in the 
            returned image.  This dataframe can be used to calculate answers to 
            questions such as how many atoms are in the image, is the image 
            mostly protein or nucleic acid, what amino acid is in the center of 
            the image, etc.
    """
    atoms = discard_atoms_outside_image(atoms, img_params)
    img = image_from_all_atoms(atoms, img_params)

    return img, atoms

def image_from_all_atoms(atoms: pl.DataFrame, img_params: ImageParams) -> Image:
    """\
    Create an voxelized representation of the given atoms.

    Arguments:
        atoms: See `image_from_atoms()`.
        img_params: See `image_from_atoms()`.

    Returns:
        See `image_from_atoms()`.

    This function is the same as `image_from_atoms()`, but without the initial 
    pruning of atoms outside the image.  This step is generally an important 
    performance optimization, but can be a waste of time if either (i) there aren't 
    many atoms that fall outside the image or (ii) the atoms were already 
    pruned by `discard_atoms_outside_image()` prior to calling this function.
    """
    if __debug__:
        _check_channels(atoms, img_params.channels)
        _check_max_radius_A(atoms, img_params.max_radius_A)

    if 'occupancy' not in atoms:
        atoms = atoms.with_columns(occupancy=1.0)

    img = _make_empty_image(img_params)

    # If the input dataframe is in the right format, the casts will be no-ops 
    # and the numpy conversions won't perform any copies.  However, this isn't 
    # generally the case.  Data types can depend on the library used to load 
    # the atoms, and filtering operations can prevent no-copy numpy conversions 
    # unless the dataframe is rechunked afterwards.
    _add_atoms_to_image(
            img,
            img_params.grid,
            atoms['x'].cast(pl.Float64).to_numpy(),
            atoms['y'].cast(pl.Float64).to_numpy(),
            atoms['z'].cast(pl.Float64).to_numpy(),
            atoms['radius_A'].cast(pl.Float64).to_numpy(),
            atoms['channels'].list.explode().cast(pl.Int64).to_numpy(),
            atoms['channels'].list.len().to_numpy(),  # no need for cast; len() always returns pl.Int32
            atoms['occupancy'].cast(pl.Float64).to_numpy(),
            img_params.fill_algorithm,
            img_params.agg_algorithm,
    )

    return img
        
def discard_atoms_outside_image(atoms: pl.DataFrame, img_params: ImageParams):
    """\
    Return only those atoms that will be present in the image.

    Arguments:
        atoms:
            See `image_from_atoms()`, but only the *x*, *y*, and *z* columns 
            are required.  If the *radius_A* column is specified, it will be 
            used.  Otherwise, *img_params.max_radius_A* must be specified, and 
            every atom will be assumed to have that radius.

        img_params:
            See `image_from_atoms()`.

    Returns:
        A copy of the atoms dataframe containing only those rows corresponding 
        to atoms that have some overlap with the image.  No changes are made to 
        any of the rows that are returned.

    The primary reason to use this function is to save a little bit of time by 
    only calculating columns such as *channels* and *radius_A* for those the 
    atoms that need them.  When using `image_from_atoms()`, there's no way to 
    know which atoms will actually be part of the image.  So if any columns 
    need to be calculated, they need to be calculated for every atom.  This can 
    be wasteful for large structures, where only a small fraction of the atoms 
    participate in the final image.

    This function is typically followed by `image_from_all_atoms()`, not 
    `image_from_atoms()`, to avoid needlessly repeating this filtering step.  
    Be careful not to modify the *x*, *y*, *z*, or *radius_A* columns of the 
    filtered *atoms* dataframe produced by this function before generating the 
    image.  Doing so could move the affected atoms outside the image, meaning 
    that the dataframe would no longer be an accurate listing of the atoms 
    in the image.
    """
    grid = img_params.grid
    max_radius_A = img_params.max_radius_A

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
        channels: List[List[str]],
        *,
        drop_missing_atoms: bool = False,
) -> pl.DataFrame:
    """\
    Assign atoms to channels based on their element types.

    Arguments:
        atoms:
            A dataframe representing the atoms to voxelize.  This function 
            requires a column named *element*, which must contain element names 
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
    """\
    Add every atom for which the expression is true to the given channel.

    Arguments:
        atoms:
            A dataframe representing the atoms to voxelize.  The dataframe must 
            already have a column named *channels*, in the format expected by 
            `image_from_atoms()`.

        expr:
            A boolean expression that will be evaluated for each atom.  If the 
            expression is true, the atom will be added to the given channel.

        channel:
            The channel to add the atoms to.

    Returns:
        A copy of the input dataframe, with the *channels* column modified.
    """
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

def find_occupied_voxels(atoms: pl.DataFrame, grid: Grid):
    """
    Return a multi-dimensional slice containing only those voxels that could be 
    occupied by an atom.

    Arguments:
        atoms: See `image_from_atoms()`.
        grid: An object specifying the size and location of each voxel.

    Returns:
        A tuple of slice objects.  This tuple can be directly used to index 
        into a numpy array.

    This function can be useful for comparing a small part of a structure, e.g. 
    a single residue, to a larger image.  The returned slices can quickly get 
    rid of most of the uninteresting voxels, allowing any calculations 
    involving the remaining voxels to complete much faster.
    """
    probes = (
            atoms
            .select(
                probes=pl.concat_list([
                    pl.struct(
                        x=pl.col('x') - pl.col('radius_A'),
                        y=pl.col('y'),
                        z=pl.col('z'),
                    ),
                    pl.struct(
                        x=pl.col('x') + pl.col('radius_A'),
                        y=pl.col('y'),
                        z=pl.col('z'),
                    ),
                    pl.struct(
                        x=pl.col('x'),
                        y=pl.col('y') - pl.col('radius_A'),
                        z=pl.col('z'),
                    ),
                    pl.struct(
                        x=pl.col('x'),
                        y=pl.col('y') + pl.col('radius_A'),
                        z=pl.col('z'),
                    ),
                    pl.struct(
                        x=pl.col('x'),
                        y=pl.col('y'),
                        z=pl.col('z') - pl.col('radius_A'),
                    ),
                    pl.struct(
                        x=pl.col('x'),
                        y=pl.col('y'),
                        z=pl.col('z') + pl.col('radius_A'),
                    ),
                ])
            )
            .explode('probes')
            .unnest('probes')
            .to_numpy()
    )

    # See the comment in `get_voxel_center_coords()` for an explanation of why 
    # we need to transpose twice here.
    voxels = _find_voxels_containing_coords(grid, probes.T).T

    def get_slice(i):
        start = max(voxels[:, i].min().item(), 0)
        end = min(voxels[:, i].max().item() + 1, grid.length_voxels)
        return slice(start, end)

    return ..., get_slice(0), get_slice(1), get_slice(2)


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

class ValidationError(Exception):
    """
    Raised when errors in the input data are detected.
    """
    pass
