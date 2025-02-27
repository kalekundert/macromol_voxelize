import macromol_voxelize as mmvox
import macromol_voxelize.voxelize as _mmvox
import macromol_voxelize._voxelize as _mmvox_cpp
import numpy as np
import polars as pl
import polars.testing
import pytest
import parametrize_from_file as pff
import pickle

from macromol_voxelize._voxelize import Sphere, Atom, Grid
from macromol_dataframe.testing import coord, coords
from io import StringIO
from itertools import product
from pytest import approx

with_py = pff.Namespace()
with_math = pff.Namespace('from math import *')
with_np = pff.Namespace(with_math, 'import numpy as np')
with_mmvox = pff.Namespace('from macromol_voxelize import *')

def grid(params):
    if isinstance(params, str):
        length_voxels = int(params)
        resolution_A = 1.0
        center_A = np.zeros(3)

    else:
        params = params.copy()
        length_voxels = int(params.pop('length_voxels'))
        resolution_A = float(params.pop('resolution_A', 1.0))
        center_A = coord(params.pop('center_A', '0 0 0'))

        if params:
            raise ValueError(f"unexpected grid parameter(s): {list(params)}")

    return Grid(length_voxels, resolution_A, center_A)

def sphere(params):
    return Sphere(
            center_A=coord(params['center_A']),
            radius_A=with_math.eval(params['radius_A']),
    )

def atom(params):
    return Atom(
            sphere=sphere(params),
            channels=[int(x) for x in params['channels'].split()],
            occupancy=float(params.get('occupancy', 1)),
    )

def atoms(params):
    dtypes = {
            'channels': pl.List(pl.Int32),
            'radius_A': float,
            'x': float,
            'y': float,
            'z': float,
            'occupancy': float,
    }
    col_aliases = {
            'c': 'channels',
            'r': 'radius_A',
            'f': 'occupancy',
    }

    rows = [line.split() for line in params.splitlines()]
    header, rows = rows[0], rows[1:]
    df = (
            pl.DataFrame(rows, header, orient='row')
            .rename(lambda x: col_aliases.get(x, x))
    )

    if 'channels' in df.columns:
        df = df.with_columns(
                pl.col('channels').str.split(','),
        )

    df = df.cast({
        (K := col_aliases.get(k, k)): dtypes.get(K, str)
        for k in header
    })

    return df

def index(params):
    return np.array([int(x) for x in params.split()])

def indices(params):
    io = StringIO(params)
    indices = np.loadtxt(io, dtype=int)
    indices.shape = (1, *indices.shape)[-2:]
    return indices

def image_params(params):
    return mmvox.ImageParams(
            channels=int(params.get('channels', '1')),
            grid=grid(params['grid']),
            dtype=with_np.eval(params.get('dtype', 'np.float32')),
            max_radius_A=eval(params.get('max_radius_A', 'None')),
            fill_algorithm=fill_algorithm(params.get('fill_algorithm', 'FractionAtom')),
            agg_algorithm=agg_algorithm(params.get('agg_algorithm', 'Sum')),
    )

def fill_algorithm(params):
    fill_algorithms = {
            'OverlapA3': mmvox.FillAlgorithm.OverlapA3,
            'FractionAtom': mmvox.FillAlgorithm.FractionAtom,
            'FractionVoxel': mmvox.FillAlgorithm.FractionVoxel,
    }
    return fill_algorithms[params]

def agg_algorithm(params):
    agg_algorithms = {
            'Sum': mmvox.AggAlgorithm.Sum,
            'Max': mmvox.AggAlgorithm.Max,
    }
    return agg_algorithms[params]

def image(params):
    return {
            tuple(index(k)): with_math.eval(v)
            for k, v in params.items()
    }

def assert_images_match(actual, expected):
    axes = [range(x) for x in actual.shape]
    for i in product(*axes):
        assert actual[i] == approx(expected.get(i, 0))


@pff.parametrize(
        schema=pff.cast(
            atoms=atoms,
            radius=float,
            expected=atoms,
        ),
)
def test_set_atom_radius_A(atoms, radius, expected):
    actual = mmvox.set_atom_radius_A(atoms, radius)
    pl.testing.assert_frame_equal(actual, expected, check_column_order=False)

@pff.parametrize(
        schema=[
            pff.cast(
                atoms=atoms,
                kwargs=with_py.eval,
                expected=atoms,
            ),
            pff.defaults(
                kwargs={},
            ),
            with_mmvox.error_or('expected'),
        ]
)
def test_set_atom_channels_by_element(atoms, channels, kwargs, expected, error):
    with error:
        actual = mmvox.set_atom_channels_by_element(atoms, channels, **kwargs)
        pl.testing.assert_frame_equal(
                actual, expected,
                check_column_order=False,
                check_dtypes=False,
        )

def test_add_atom_channel_by_col():
    atoms = pl.DataFrame([
        dict(channels=[   ], a=False),
        dict(channels=[0  ], a=False),
        dict(channels=[  1], a=False),
        dict(channels=[0,1], a=False),
        dict(channels=[   ], a=True),
        dict(channels=[0  ], a=True),
        dict(channels=[  1], a=True),
        dict(channels=[0,1], a=True),
    ])
    atoms = mmvox.add_atom_channel_by_expr(atoms, 'a', 2)

    assert atoms.to_dicts() == [
        dict(channels=[     ], a=False),
        dict(channels=[0    ], a=False),
        dict(channels=[  1  ], a=False),
        dict(channels=[0,1  ], a=False),
        dict(channels=[    2], a=True),
        dict(channels=[0,  2], a=True),
        dict(channels=[  1,2], a=True),
        dict(channels=[0,1,2], a=True),
    ]

def test_add_atom_channel_by_expr():
    atoms = pl.DataFrame([
        dict(channels=[   ], a=3),
        dict(channels=[0  ], a=3),
        dict(channels=[  1], a=3),
        dict(channels=[0,1], a=3),
        dict(channels=[   ], a=4),
        dict(channels=[0  ], a=4),
        dict(channels=[  1], a=4),
        dict(channels=[0,1], a=4),
    ])
    atoms = mmvox.add_atom_channel_by_expr(atoms, pl.col('a') == 4, 2)

    assert atoms.to_dicts() == [
        dict(channels=[     ], a=3),
        dict(channels=[0    ], a=3),
        dict(channels=[  1  ], a=3),
        dict(channels=[0,1  ], a=3),
        dict(channels=[    2], a=4),
        dict(channels=[0,  2], a=4),
        dict(channels=[  1,2], a=4),
        dict(channels=[0,1,2], a=4),
    ]


@pff.parametrize(
        schema=pff.cast(
            atoms=atoms,
            img_params=image_params,
            expected=image,
        ),
)
def test_image_from_atoms(atoms, img_params, expected):
    img, _ = mmvox.image_from_atoms(atoms, img_params)
    assert_images_match(img, expected)
    assert img.dtype == img_params.dtype

@pytest.mark.xfail
def test_image_from_atoms_chunks():
    # 2024/10/10: I marked this test as xfail because it's flaky on the CI 
    # server.  How exactly the data frame gets chunked seems to depend on 
    # something outside of this test, perhaps the version of polars/arrow being 
    # used or something like that.

    # This test makes much more sense in the context of passing the Arrow table 
    # directly to the C++ code.  I'm not doing that anymore, but this still 
    # seems like a reasonable test case.  It checks that copies can occur when 
    # they have to.
    
    x = pl.Series("x", [-0.5, 0.5])
    x.append(pl.Series([-0.5, 0.5]))
    x.append(pl.Series([-0.5, 0.5]))

    y = pl.Series("y", [-0.5, -0.5, 0.5])
    y.append(pl.Series([0.5, -0.5, -0.5]))

    z = pl.Series("z", [-0.5, -0.5, -0.5, -0.5, 0.5, 0.5])

    atoms = (
            pl.DataFrame([x, y, z])
            .with_columns(
                radius_A=0.49,
                channels=[0],
                occupancy=1.0
            )
    )
    assert atoms.n_chunks('all') == [3, 2, 1, 1, 1, 1]

    img_params = mmvox.ImageParams(
            channels=1,
            grid=Grid(length_voxels=2, resolution_A=1),
    )
    img, _ = mmvox.image_from_atoms(atoms, img_params)

    expected = {
            (0,0,0,0): 1,
            (0,1,0,0): 1,
            (0,0,1,0): 1,
            (0,1,1,0): 1,
            (0,0,0,1): 1,
            (0,1,0,1): 1,
    }

    assert_images_match(img, expected)

def test_make_empty_image():
    img_params = mmvox.ImageParams(
            channels=2,
            grid=Grid(
                length_voxels=3,
                resolution_A=1,         # not relevant
                center_A=np.zeros(3),   # not relevant
            ),
    )
    np.testing.assert_array_equal(
            _mmvox._make_empty_image(img_params),
            np.zeros((2, 3, 3, 3)),
            verbose=True,
    )

@pff.parametrize(
        schema=[
            pff.cast(
                atoms=atoms,
                img_params=image_params,
                expected=atoms,
            ),
        ],
)
def test_discard_atoms_outside_image(atoms, img_params, expected):
    actual = mmvox.discard_atoms_outside_image(atoms, img_params)
    pl.testing.assert_frame_equal(actual, expected)

@pff.parametrize(
        schema=pff.cast(
            atoms=atoms,
            num_channels=int,
            error=with_mmvox.error,
        )
)
def test_check_channels(atoms, num_channels, error):
    with error:
        _mmvox._check_channels(atoms, num_channels)

@pff.parametrize(
        schema=pff.cast(
            atoms=atoms,
            max_radius_A=float,
            error=with_mmvox.error,
        )
)
def test_check_max_radius_A(atoms, max_radius_A, error):
    with error:
        _mmvox._check_max_radius_A(atoms, max_radius_A)


def test_add_atoms_to_image_err_no_copy():
    atoms = pl.DataFrame([
        dict(x=0.0, y=0.0, z=0.0, radius_A=0.5, channels=[0], occupancy=1.0),
    ])
    grid = Grid(length_voxels=2, resolution_A=1)

    # Integer data types are not supported.  Instead of silently making a copy, 
    # the binding code should notice the discrepancy and complain.
    img = np.zeros((2, 3, 3, 3), dtype=np.int64)

    with pytest.raises(TypeError):
        _mmvox._add_atoms_to_image(
                img, grid,
                atoms['x'].to_numpy(),
                atoms['y'].to_numpy(),
                atoms['z'].to_numpy(),
                atoms['radius_A'].to_numpy(),
                atoms['channels'].list.explode().cast(pl.Int32).to_numpy(),
                atoms['channels'].list.len().to_numpy(),
                atoms['occupancy'].to_numpy(),
                _mmvox_cpp.FillAlgorithm.FractionAtom,
                _mmvox_cpp.AggAlgorithm.Sum,
        )

@pff.parametrize(
        schema=pff.cast(img_params=image_params, atom=atom, expected=image)
)
def test_add_atom_to_image(img_params, atom, expected):
    img = _mmvox._make_empty_image(img_params)
    _mmvox_cpp._add_atom_to_image(
            img,
            img_params.grid,
            atom,
            img_params.fill_algorithm,
            img_params.agg_algorithm,
    )
    assert_images_match(img, expected)

def test_add_atom_to_image_err_no_copy():
    grid = Grid(
            length_voxels=3,
            resolution_A=1,
    )
    atom = Atom(
            sphere=Sphere(
                center_A=np.zeros(3),
                radius_A=1,
            ),
            channels=[0],
            occupancy=1,
    )

    # Integer data types are not supported.  Instead of silently making a copy, 
    # the binding code should notice the discrepancy and complain.
    img = np.zeros((2, 3, 3, 3), dtype=np.int64)

    with pytest.raises(TypeError):
        _mmvox_cpp._add_atom_to_image(img, grid, atom)

@pff.parametrize(
        schema=pff.cast(
            grid=grid,
            sphere=sphere,
            expected=pff.cast(
                min_index=index,
                max_index=index,
            ),
        ),
)
def test_find_voxels_possibly_contacting_sphere(grid, sphere, expected):
    voxels = _mmvox_cpp._find_voxels_possibly_contacting_sphere(grid, sphere)
    voxel_tuples = {
            tuple(x)
            for x in voxels.T
    }

    if expected == 'empty':
        expected_tuples = set()
    else:
        axes = [
                range(expected['min_index'][i], expected['max_index'][i] + 1)
                for i in range(3)
        ]
        expected_tuples = {
                (i, j, k)
                for i, j, k in product(*axes)
        }

    assert voxel_tuples >= expected_tuples

@pff.parametrize(
        key=['test_get_voxel_center_coords', 'test_find_voxels_containing_coords'],
        schema=pff.cast(grid=grid, coords=coords, voxels=indices),
)
def test_find_voxels_containing_coords(grid, coords, voxels):
    np.testing.assert_array_equal(
            _mmvox_cpp._find_voxels_containing_coords(grid, coords.T),
            voxels.T,
            verbose=True,
    )

@pff.parametrize(
        schema=pff.cast(grid=grid, voxels=indices, expected=indices),
)
def test_discard_voxels_outside_image(grid, voxels, expected):
    np.testing.assert_array_equal(
            _mmvox_cpp._discard_voxels_outside_image(grid, voxels.T),
            expected.reshape(-1, 3).T,
    )

@pff.parametrize(
        schema=pff.cast(grid=grid, voxels=indices, coords=coords),
)
def test_get_voxel_center_coords(grid, voxels, coords):
    actual = mmvox.get_voxel_center_coords(grid, voxels)
    assert actual == approx(coords)

@pff.parametrize(
        schema=pff.cast(
            atoms=atoms,
            img_params=image_params,
            expected=with_py.eval,
        ),
)
def test_find_occupied_voxels(atoms, img_params, expected):
    ijk = mmvox.find_occupied_voxels(atoms, img_params.grid)
    assert ijk == tuple(expected)

    # Check that all voxels outside the occupied region are zero.  The best way 
    # I could think to do this is to set the occupied region to zero, then 
    # check that the whole image is zero.  Note that some test cases have atoms 
    # that are outside the image, so we can't really conclude anything about 
    # the values in the occupied region itself.
    img = mmvox.image_from_all_atoms(atoms, img_params)
    img[ijk] = 0
    assert np.sum(img) == 0

def test_sphere_attrs():
    s = Sphere(
            center_A=np.array([1,2,3]),
            radius_A=4,
    )
    assert s.center_A == approx([1,2,3])
    assert s.radius_A == 4

    # https://www.omnicalculator.com/math/sphere-volume
    assert s.volume_A3 == approx(268.1, abs=0.1)

def test_sphere_repr():
    s = Sphere(
            center_A=np.array([1,2,3]),
            radius_A=4,
    )
    s_repr = eval(repr(s))

    np.testing.assert_array_equal(s_repr.center_A, [1,2,3])
    assert s_repr.radius_A == 4

def test_sphere_pickle():
    s = Sphere(
            center_A=np.array([1,2,3]),
            radius_A=4,
    )
    s_pickle = pickle.loads(pickle.dumps(s))

    np.testing.assert_array_equal(s_pickle.center_A, [1,2,3])
    assert s_pickle.radius_A == 4


def test_grid_attrs():
    g = Grid(
            center_A=np.array([1,2,3]),
            length_voxels=4,
            resolution_A=0.5,
    )
    assert g.center_A == approx([1,2,3])
    assert g.length_voxels == 4
    assert g.resolution_A == 0.5

def test_grid_repr():
    g = Grid(
            center_A=np.array([1,2,3]),
            length_voxels=4,
            resolution_A=0.5,
    )
    g_repr = eval(repr(g))

    np.testing.assert_array_equal(g_repr.center_A, [1,2,3])
    assert g_repr.length_voxels == 4
    assert g_repr.resolution_A == 0.5

def test_grid_pickle():
    g = Grid(
            center_A=np.array([1,2,3]),
            length_voxels=4,
            resolution_A=0.5,
    )
    g_pickle = pickle.loads(pickle.dumps(g))

    np.testing.assert_array_equal(g_pickle.center_A, [1,2,3])
    assert g_pickle.length_voxels == 4
    assert g_pickle.resolution_A == 0.5


def test_atom_attrs():
    a = Atom(
            sphere=Sphere(
                center_A=np.array([1,2,3]),
                radius_A=4,
            ),
            channels=[0],
            occupancy=0.5,
    )
    assert a.sphere.center_A == approx([1,2,3])
    assert a.sphere.radius_A == 4
    assert a.channels == [0]
    assert a.occupancy == 0.5

def test_atom_repr():
    a = Atom(
            sphere=Sphere(
                center_A=np.array([1,2,3]),
                radius_A=4,
            ),
            channels=[0],
            occupancy=0.5,
    )
    a_repr = eval(repr(a))

    np.testing.assert_array_equal(a_repr.sphere.center_A, [1,2,3])
    assert a_repr.sphere.radius_A == 4
    assert a_repr.channels == [0]
    assert a_repr.occupancy == 0.5

def test_atom_pickle():
    a = Atom(
            sphere=Sphere(
                center_A=np.array([1,2,3]),
                radius_A=4,
            ),
            channels=[0],
            occupancy=0.5,
    )
    a_pickle = pickle.loads(pickle.dumps(a))

    np.testing.assert_array_equal(a_pickle.sphere.center_A, [1,2,3])
    assert a_pickle.sphere.radius_A == 4
    assert a_pickle.channels == [0]
    assert a_pickle.occupancy == 0.5
