# CHANGELOG



## v0.1.0 (2024-05-01)

### Chore

* chore: debug automated releases ([`0bc301a`](https://github.com/kalekundert/macromol_voxelize/commit/0bc301a4526f35672a792ade7eeff872637b64be))

* chore: configure automated releases ([`37f1d47`](https://github.com/kalekundert/macromol_voxelize/commit/37f1d47dc1c7a4a84845ea56bed423f5b343bb90))

* chore: apply cookiecutter ([`6b12ba3`](https://github.com/kalekundert/macromol_voxelize/commit/6b12ba363c8b486e9258b2c4358c046ab2ee42b8))

### Documentation

* docs: write a README file ([`a04cd22`](https://github.com/kalekundert/macromol_voxelize/commit/a04cd226698728db1d7da5de10668e3e19d8ec51))

### Feature

* feat: use pymol to visualize voxels ([`f81e445`](https://github.com/kalekundert/macromol_voxelize/commit/f81e44530c1aea75fc587e4c1d0a2015d78ceb03))

* feat: expect channel and radius columns

Previously, I envisioned the main input to the voxelization method being
the raw atoms dataframe; i.e. just columns that generically describe
macromolecules.  Alongside this dataframe, there would also have to be
lists/dicts/functions describing how to pick radii and channels for each
atom.

I realized that this approach was inflexible.  Better is to expect a
dataframe than specifies all the information I need, including the
channels and radii.  This simplifies the voxelization API, and gives the
user the power to work out these parameters in whatever complicated ways
they want.  The library can also provide some functions to calculate
some common radius/channel mappings. ([`66c74a5`](https://github.com/kalekundert/macromol_voxelize/commit/66c74a55a6cde1203be12df1fbe88b9f09c61820))

* feat: initial implementation

Most of the code is copied from atompaint, but I did modify it to use
polars instead of pandas. ([`30a77fd`](https://github.com/kalekundert/macromol_voxelize/commit/30a77fdc914c901064920ba2df4d706eb6abd743))

### Fix

* fix: require channel patterns to match full element names ([`e7dc291`](https://github.com/kalekundert/macromol_voxelize/commit/e7dc29164729dcf4dad4d6c8fb282a0e069b47b3))
