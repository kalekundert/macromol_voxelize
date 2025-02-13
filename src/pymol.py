import pymol
import polars as pl
import numpy as np
import macromol_dataframe as mmdf
import mixbox

from pymol import cmd
from pymol.cgo import (
        ALPHA, BEGIN, COLOR, CONE, CYLINDER, END, LINES, NORMAL, TRIANGLE_FAN,
        VERTEX,
)
from pathlib import Path
from itertools import product, chain, repeat
from more_itertools import take
from pipeline_func import f

from macromol_voxelize import (
        ImageParams, Grid, image_from_atoms, set_atom_radius_A,
        set_atom_channels_by_element, get_voxel_center_coords,
)

def voxelize(
        center_sele=None,
        center_A=None,
        all_sele='all',
        length_voxels=35,
        resolution_A=1,
        channels='C,N,O',
        element_radius_A=None,
        outline=False,
        state=-1,
        sele_name='within_img',
        img_name='voxels',
        outline_name='outline',
        out_path=None,
):
    """\
DESCRIPTION

    Create a 3D image of the atoms in the specified selection.
    
USAGE

    voxelize [center_sele [, center_A [, all_sele [, length_voxels
        [, resolution_A [, channels [, element_radius_A [, outline [, state
        [, sele_name [, img_name [, outline_name [, out_path]]]]]]]]]]]]

ARGUMENTS

    center_sele = string: The atoms to center the image on.  Specifically, the 
    image will be centered on the center of mass of the given selection.

    center_A = float vector: The center of the image, in Angstroms.  This 
    option supersedes the `center_sele` option, if both are given.

    all_sele = string: The atoms to include in the image. {default: 'all'}

    length_voxels = int: The length of each side of the image, in voxels.  Note 
    that the image will always be a cube. {default: 35}

    resolution_A = float: The size of each voxel, in Angstroms. {default: 1}

    channels = string list: A comma-separated list of element names.  Each 
    entry in this list will correspond to a channel in the image. {default: 
    'C,N,O'}

    element_radius_A = float: The radius of each atom, in Angstroms.

    outline = bool: Whether to render an outline of the image, "yes" or "no". 
    {default: no}

    state = int: Which state to render. {default: -1}

    sele_name = string: The name to use when creating a selection of all the 
    atoms that are actually contained within the image. {default: 'within_img'}

    img_name = string: The name to use for the image object that will be 
    created by this command {default: 'voxels'}

    outline_name = string: The name to use for the outline object that will be 
    created by this command, if the `outline` option is enabled. {default: outline}

    out_path = string: The path to save the image to.  If not given, the image 
    will not be saved.

EXAMPLES

    voxelize center_A=[0, 0, 0], length_voxels=35, resolution_A=1
    
"""
    atoms = mmdf.from_pymol(all_sele, state)
    length_voxels = int(length_voxels)
    resolution_A = float(resolution_A)
    channels = parse_channels(channels)
    element_radius_A = parse_element_radius_A(element_radius_A, resolution_A)
    state = int(state)

    if center_A is not None:
        center_A = np.array(eval(center_A))
    else:
        center_A = np.array(cmd.centerofmass(center_sele or all_sele, state))

    atoms = (
            atoms
            | f(set_atom_channels_by_element, channels)
            | f(set_atom_radius_A, element_radius_A)
    )
    img_params = ImageParams(
            grid=Grid(
                length_voxels=length_voxels,
                resolution_A=resolution_A,
                center_A=center_A,
            ),
            channels=len(channels),
    )
    select_view(
            sele_name,
            all_sele,
            img_params.grid,
    )
    render_view(
            obj_names=dict(
                voxels=img_name,
                outline=outline_name,
            ),
            atoms_i=atoms,
            img_params=img_params,
            channel_colors=pick_channel_colors(sele_name, channels),
            outline=outline,
            out_path=out_path,
    )

pymol.cmd.extend('voxelize', voxelize)
cmd.auto_arg[0]['voxelize'] = cmd.auto_arg[0]['zoom']

def load_voxels(
        img_path,
        resolution_A=1,
        channel=None,
        img_name=None,
        outline=False,
        outline_name='outline',
        color_scheme='CNOPS',
        scale_alpha='no',
        batch_index=-1,
):
    """\
DESCRIPTION

    Render the image contained in the given file.
    
USAGE

    load_voxels img_path [, resolution_A [, channel [, img_name [, outline
        [, outline_name [, color_scheme [, scale_alpha [, batch_index]]]]]]]

ARGUMENTS

    img_path = string: The path to the image file.  To create an image file, 
    call `numpy.save(path, image)` on the image made by `image_from_atoms()`.

    resolution_A = float: The size of each voxel, in Angstroms. {default: 1}

    channel = int: The channel to render.  By default, all channels are 
    rendered.

    img_name = string: The name to use for the image object that will be 
    created by this command.  The default is the "stem" of the image path.

    outline = bool: Whether to render an outline of the image, "yes" or "no". 
    {default: no}

    outline_name = string: The name to use for the outline object that will be 
    created by this command, if the `outline` option is enabled. {default: outline}

    color_scheme = string: The colors to use for each channel.  The default is 
    'CNOPS', which uses the carbon color for the first channel, nitrogen color 
    for the second channel, etc.  If there are more than 5 channels, the 
    remaining channels will all be white.  You can also provide a colon- 
    separated list of pymol colors.

    scale_alpha = bool: If true, scale the image so that the maximum voxel has 
    a value of 1.

    batch_index = int: Which image to render (indexing from 0), if the given 
    image path contains multiple images.

EXAMPLES

    load_voxels path/to/image.npy

"""
    img_path = Path(img_path)
    img = np.load(img_path)

    if len(img.shape) not in [4, 5]:
        raise ValueError(f"expected an image of dimension [B,] C, W, H, D; got {len(img.shape)} dimensions")

    w, h, d = img.shape[-3:]
    if w != h or h != d:
        raise ValueError(f"inconsistent image dimensions: {w}, {h}, {d}")

    if channel is not None:
        img = img[..., [int(channel)], :, :, :]
        colors = [(1, 1, 1)]

    else:
        c = img.shape[-4]

        if color_scheme == 'CNOPS':
            color_names_inf = chain(
                    ['carbon', 'nitrogen', 'oxygen', 'phosphorus', 'sulfur'],
                    repeat('white'),
            )
            color_names = list(take(c, color_names_inf))

        else:
            color_names = color_scheme.split(':')
            if len(color_names) != c:
                raise ValueError(f"expected {c} colors, got {len(color_names)}")

        colors = []

        for i, k in enumerate(color_names):
            print(f"channel {i}: {k}")
            colors.append(cmd.get_color_tuple(k))

    render_image(
            obj_names=dict(
                voxels=img_name or img_path.stem,
                outline=outline_name,
            ),
            img=img,
            grid=Grid(
                length_voxels=d,
                resolution_A=float(resolution_A),
            ),
            channel_colors=colors,
            outline=outline,
            scale_alpha=parse_bool(scale_alpha),
            batch_index=int(batch_index),
    )

pymol.cmd.extend('load_voxels', load_voxels)
cmd.auto_arg[0]['load_voxels'] = [
        lambda: cmd.Shortcut([p.name for p in Path.cwd().glob('*.npy')]),
        'image path (*.npy)',
        '',
]

def render_view(
        *,
        obj_names,
        atoms_i,
        img_params,
        channel_colors,
        axes=False,
        outline=False,
        img=True,
        frame_ix=None,
        scale_alpha=False,
        out_path=None,
):
    if frame_ix is not None:
        atoms_x = mmdf.transform_atom_coords(atoms_i, frame_ix)
        frame_xi = mmdf.invert_coord_frame(frame_ix)
    else:
        atoms_x = atoms_i
        frame_xi = None

    if img:
        img, _ = image_from_atoms(atoms_x, img_params)
        if out_path:
            np.save(out_path, img)
    else:
        img = None

    render_image(
            obj_names=obj_names,
            img=img,
            grid=img_params.grid,
            channel_colors=channel_colors,
            axes=axes,
            outline=outline,
            frame_xi=frame_xi,
            scale_alpha=scale_alpha,
    )

def render_image(
        *,
        obj_names,
        img,
        grid,
        channel_colors,
        axes=False,
        outline=False,
        frame_xi=None,
        scale_alpha=False,
        batch_index=-1,
):
    view = cmd.get_view()

    # Important to render the axes before the voxels.  I don't know why, but if 
    # the voxels are rendered first, PyMOL regards them as opaque (regardless 
    # of the `transparency_mode` setting.
    if axes:
        ax = cgo_axes()
        cmd.delete(obj_names['axes'])
        cmd.load_cgo(ax, obj_names['axes'])

    if outline:
        edges = cgo_cube_edges(grid.center_A, grid.length_A, outline)
        cmd.delete(obj_names['outline'])
        cmd.load_cgo(edges, obj_names['outline'])

    if img is not None:
        # If `transparency_mode` is disabled (which is the default), CGOs will 
        # be opaque no matter what.
        cmd.set('transparency_mode', 1)

        if len(img.shape) not in [4, 5]:
            raise ValueError(f"expected an image of dimension [B,] C, W, H, D; got {len(img.shape)} dimensions")

        if batch_index != -1:
            if len(img.shape) == 4:
                raise ValueError("requested batch index {batch_index}, but given image has no batch dimension")
            img = img[batch_index]

        if len(img.shape) == 4:
            img = img[np.newaxis, ...]

        if scale_alpha:
            img = img / img.max()

        cmd.delete(obj_names['voxels'])

        for b in range(img.shape[0]):
            voxels = cgo_voxels(img[b], grid, channel_colors)
            cmd.load_cgo(voxels, obj_names['voxels'], state=b+1)

    if frame_xi is not None:
        for obj in obj_names.values():
            frame_1d = frame_xi.flatten().tolist()
            cmd.set_object_ttt(obj, frame_1d, state=-1)

    cmd.set_view(view)

def select_view(name, sele, grid, frame_ix=None):
    indices = []
    cmd.iterate(
            selection=sele,
            expression='indices.append(index)',
            space=locals(),
    )

    coords_i = np.zeros((len(indices), 4))
    i_from_index = {x: i for i, x in enumerate(indices)}
    cmd.iterate_state(
            selection=sele,
            expression='coords_i[i_from_index[index]] = (x, y, z, 1)',
            space=locals(),
            state=1,
    )

    if frame_ix is not None:
        coords_x = mmdf.transform_coords(coords_i, frame_ix)
    else:
        coords_x = coords_i

    coords_x = coords_x[:,:3] - grid.center_A
    half_len = grid.length_A / 2
    within_grid = np.logical_and(
            coords_x >= -half_len,
            coords_x <= half_len,
    ).all(axis=1)

    cmd.alter(sele, 'b = within_grid[i_from_index[index]]', space=locals())
    cmd.select(name, 'b = 1')

def parse_channels(channels_str):
    return [[x] for x in channels_str.split(',') + ['*']]

def parse_element_radius_A(element_radius_A, resolution_A):
    if element_radius_A is None:
        return resolution_A / 2
    else:
        return float(element_radius_A)

def parse_bool(x: str) -> bool:
    return {
            'yes': True,  'true': True,   'on': True,    '1': True,
            'no': False,  'false': False, 'off': False,  '0': False,
    }[x.lower()]

def pick_channel_colors(sele, channels):
    elem_colors = []
    cmd.iterate(
            sele,
            'elem_colors.append(dict(element=elem, color=color, occupancy=q))',
            space=locals(),
    )

    elem_colors = pl.DataFrame(elem_colors)
    color_channels = set_atom_channels_by_element(
            elem_colors,
            channels,
    )
    most_common_colors = dict(
            color_channels
            .explode('channels')
            .group_by('channels', 'color')
            .agg(pl.col('occupancy').sum())
            .group_by('channels')
            .agg(pl.all().sort_by('occupancy').last())
            .select('channels', 'color')
            .iter_rows()
    )

    colors = []
    for channel in range(len(channels)):
        try:
            color_i = most_common_colors[channel]
            rgb = cmd.get_color_tuple(color_i)
        except KeyError:
            rgb = (1, 1, 1)

        colors.append(rgb)

    return colors

def cgo_voxels(img, grid, channel_colors=None):
    c, w, h, d = img.shape
    voxels = []

    alpha = get_alpha(img)
    face_masks = pick_faces(alpha)

    if channel_colors is None:
        from matplotlib.cm import tab10
        channel_colors = tab10.colors[:c]
    if len(channel_colors) != c:
        raise ValueError(f"Image has {c} channels, but {len(channel_colors)} colors were specified")

    for i, j, k in product(range(w), range(h), range(d)):
        if alpha[i, j, k] == 0:
            continue

        voxels += cgo_cube(
                get_voxel_center_coords(grid, np.array([i, j, k])),
                grid.resolution_A,
                color=mix_colors(channel_colors, img[:, i, j, k]),
                alpha=alpha[i, j, k],
                face_mask=face_masks[:, i, j, k],
        )

    return voxels

def cgo_cube(center, length, color=(1, 1, 1), alpha=1.0, face_mask=6 * (1,)):
    # The starting point for this function came from the PyMOL wiki:
    #
    # https://pymolwiki.org/index.php/Cubes
    #
    # However, this starting point (i) didn't support color or transparency and 
    # (ii) had some bugs relating to surface normals.

    verts = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
    ])
    verts = length * (verts - 0.5) + np.array(center)

    # The order in which the vertices are specified is important: it determines 
    # which direction the triangle faces.  Specifically, a triangle is facing 
    # the camera when its vertices appear in counter-clockwise order.
    #
    # https://stackoverflow.com/questions/8142388/in-what-order-should-i-send-my-vertices-to-opengl-for-culling#8142461
    #
    # Cube:
    #   2───6    y
    #  ╱│  ╱│    │
    # 3─┼─7 │    │
    # │ 0─┼─4    o───x
    # │╱  │╱    ╱
    # 1───5    z 
    #
    # Faces:
    #   x     -x      y     -y      z     -z
    # 7───6  2───3  2───6  1───5  3───7  6───2
    # │   │  │   │  │   │  │   │  │   │  │   │
    # │   │  │   │  │   │  │   │  │   │  │   │
    # 5───4  0───1  3───7  0───4  1───5  4───0
    #
    # In all of the triangle fans below, I'll start with the lower-left vertex 
    # (e.g. 0 for the -x face) and continue counter-clockwise.

    def face(normal, indices):
        return [
                BEGIN, TRIANGLE_FAN,
                ALPHA, alpha,
                COLOR, *color,
                NORMAL, *normal,
                VERTEX, *verts[indices[0]],
                VERTEX, *verts[indices[1]],
                VERTEX, *verts[indices[2]],
                VERTEX, *verts[indices[3]],
                END,
        ]

    faces = []
    x, y, z = np.eye(3)

    if face_mask[0]: faces += face(+x, [5, 4, 6, 7])
    if face_mask[1]: faces += face(-x, [0, 1, 3, 2])
    if face_mask[2]: faces += face(+y, [3, 7, 6, 2])
    if face_mask[3]: faces += face(-y, [0, 4, 5, 1])
    if face_mask[4]: faces += face(+z, [1, 5, 7, 3])
    if face_mask[5]: faces += face(-z, [4, 0, 2, 6])

    return faces

def cgo_cube_edges(center, length, color=(1, 1, 1)):
    if color and not isinstance(color, tuple):
        color = (1, 1, 0)

    verts = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
    ])
    verts = length * (verts - 0.5) + np.array(center)

    #   2───6
    #  ╱│  ╱│
    # 3─┼─7 │
    # │ 0─┼─4
    # │╱  │╱
    # 1───5

    edges = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7),
    ]

    cube = [
            BEGIN, LINES,
            COLOR, *color,
    ]

    for i, j in edges:
        cube += [
                VERTEX, *verts[i],
                VERTEX, *verts[j],
        ]

    cube += [
            END,
    ]

    return cube

def cgo_axes():
    w = 0.06        # cylinder width 
    l1 = 0.75       # cylinder length
    l2 = l1 + 0.25  # cylinder + cone length
    d = w * 1.618   # cone base diameter

    origin = np.zeros(3)
    x, y, z = np.eye(3)
    r, g, b = np.eye(3)

    return [
            CYLINDER, *origin, *(l1 * x), w, *r, *r,
            CYLINDER, *origin, *(l1 * y), w, *g, *g,
            CYLINDER, *origin, *(l1 * z), w, *b, *b,
            CONE, *(l1 * x), *(l2 * x), d, 0, *r, *r, 1, 1,
            CONE, *(l1 * y), *(l2 * y), d, 0, *g, *g, 1, 1,
            CONE, *(l1 * z), *(l2 * z), d, 0, *b, *b, 1, 1,
    ]

def get_alpha(img):
    img = np.sum(img, axis=0)
    return np.clip(img, 0, 1)

def pick_faces(img):
    face_masks = np.ones((6, *img.shape), dtype=bool)

    face_masks[0, :-1] = img[:-1] > img[1:]
    face_masks[1, 1:] = img[1:] > img[:-1]
    face_masks[2, :, :-1] = img[:, :-1] > img[:, 1:]
    face_masks[3, :, 1:] = img[:, 1:] > img[:, :-1]
    face_masks[4, :, :, :-1] = img[:, :, :-1] > img[:, :, 1:]
    face_masks[5, :, :, 1:] = img[:, :, 1:] > img[:, :, :-1]

    return face_masks

def mix_colors(colors, weights=None):
    if weights is None:
        weights = np.ones(len(colors))

    weights = np.array(weights).reshape(-1, 1)
    ratios = weights / np.sum(weights)

    latent_in = np.array([
            mixbox.float_rgb_to_latent(x)
            for x in colors
    ])
    latent_out = np.sum(latent_in * ratios, axis=0)

    return mixbox.latent_to_float_rgb(latent_out)

