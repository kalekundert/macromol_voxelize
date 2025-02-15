**************
PyMOL Commands
**************

This library includes two commands for visualizing 3D macromolecular images in 
PyMOL:

.. list-table::

  * - :doc:`pymol/voxelize`
    - Generate a 3D image of a macromolecule in PyMOL.
  * - :doc:`pymol/load_voxels`
    - Load a 3D image of a macromolecule into PyMOL.

In order to use these commands, the ``macromol_voxelize`` package must be 
installed into the same version of python used by PyMOL.  The details of how to 
do this depend on how PyMOL was installed.  You can always find the path to the 
relevant python installation by running the following command in PyMOL::

  PyMOL> import sys; print(sys.exec_prefix)

However, be aware that if the resulting path is managed by the operating system 
(i.e. if it starts with ``/usr`` or something similar), then you will need root 
access to install ``macromol_voxelize``.  Even if you have root access, you 
should probably not do this.  Manually installing packages into OS-managed 
python installations can cause dependency conflicts in other packages installed 
by the OS.  Instead, you should make a virtual environment, then install both 
``PyMOL`` and ``macromol_voxelize`` into that environment.  Note that 
installing PyMOL into a virtual environment will require compiling it from 
source.  The details of how to do that will depend on whether you're using 
open-source or incentive PyMOL, and are unfortunately beyond the scope of this 
documentation.

.. toctree::
   :hidden:

   pymol/voxelize
   pymol/load_voxels
