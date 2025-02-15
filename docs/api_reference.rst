*************
API Reference
*************

..
   When sphinx 8.2 comes out, add `:no-index-entry:` to the `.. autoclass` and 
   `..  autofunction`` directives in the autosummary templates.

.. rubric:: Data structures

.. autosummary::
   :toctree: api

   macromol_voxelize.ImageParams
   macromol_voxelize.Grid
   macromol_voxelize.Image

.. rubric:: Functions

.. autosummary::
   :toctree: api

   macromol_voxelize.image_from_atoms
   macromol_voxelize.image_from_all_atoms
   macromol_voxelize.discard_atoms_outside_image
   macromol_voxelize.set_atom_radius_A
   macromol_voxelize.set_atom_channels_by_element
   macromol_voxelize.add_atom_channel_by_expr
   macromol_voxelize.get_voxel_center_coords
   macromol_voxelize.find_occupied_voxels

.. rubric:: Exceptions

.. autosummary::
   :toctree: api

   macromol_voxelize.ValidationError
