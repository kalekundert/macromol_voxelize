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
from itertools import product
from more_itertools import take
from pipeline_func import f

from macromol_voxelize import (
        ImageParams, Grid, image_from_atoms, set_atom_radius_A,
        set_atom_channels_by_element, get_voxel_center_coords,
)

def voxelize(
        center_sele=None,
        all_sele='all',
        length_voxels=21,
        resolution_A=0.75,
        channels='C,N,O',
        element_radius_A=None,
        outline=False,
        state=-1,
        sele_name='within_img',
        obj_name='voxels',
        outline_name='outline',
):
    atoms = mmdf.from_pymol(all_sele, state)
    length_voxels = int(length_voxels)
    resolution_A = float(resolution_A)
    center_A = np.array(cmd.centerofmass(center_sele or all_sele, state))
    channels = parse_channels(channels)
    element_radius_A = parse_element_radius_A(element_radius_A, resolution_A)
    state = int(state)

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
                voxels=obj_name,
                outline=outline_name,
            ),
            atoms_i=atoms,
            img_params=img_params,
            channel_colors=pick_channel_colors(sele_name, channels),
            outline=outline,
    )

pymol.cmd.extend('voxelize', voxelize)
cmd.auto_arg[0]['voxelize'] = cmd.auto_arg[0]['zoom']

def load_voxels(
        img_path,
        resolution_A,
        channel=None,
        obj_name='voxels',
        outline_name='outline',
):
    img = np.load(img_path)

    c, w, h, d = img.shape
    if w != h or h != d:
        raise ValueError(f"inconsistent image dimensions: {w}, {h}, {d}")

    if channel is not None:
        img = img[[int(channel)]]
        colors = [(1, 1, 1)]
    else:
        from matplotlib.colors import TABLEAU_COLORS, hex2color
        colors = []
        n = img.shape[0]

        for i, k in enumerate(take(n, TABLEAU_COLORS)):
            print(f"channel {i}: {k[4:]}")
            colors.append(hex2color(TABLEAU_COLORS[k]))

    render_image(
            obj_names=dict(
                voxels=obj_name,
                outline=outline_name,
            ),
            img=img,
            grid=Grid(
                length_voxels=d,
                resolution_A=float(resolution_A),
            ),
            channel_colors=colors,
    )

pymol.cmd.extend('load_voxels', load_voxels)

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
        state=-1,
):
    if frame_ix is not None:
        atoms_x = mmdf.transform_atom_coords(atoms_i, frame_ix)
        frame_xi = mmdf.invert_coord_frame(frame_ix)
    else:
        atoms_x = atoms_i
        frame_xi = None

    if img:
        img = image_from_atoms(atoms_x, img_params)
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
            state=state,
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
        state=-1,
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

        if scale_alpha:
            img = img / img.max()

        voxels = cgo_voxels(img, grid, channel_colors)
        cmd.delete(obj_names['voxels'])
        cmd.load_cgo(voxels, obj_names['voxels'])

    if frame_xi is not None:
        for obj in obj_names.values():
            frame_1d = frame_xi.flatten().tolist()
            cmd.set_object_ttt(obj, frame_1d, state)

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

def pick_channel_colors(sele, channels):
    elem_colors = []
    cmd.iterate(
            sele,
            'elem_colors.append(dict(element=elem, color=color))',
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
            .len()
            .group_by('channels')
            .agg(pl.all().sort_by('len').last())
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

