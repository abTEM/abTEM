Cut a Release
=============

**Preparation**
- Bump version in `abtem/_version.py`

**Tag and release**
:warning: this is a point of no return point :warning:
- push tag (`x.y.z`) to the upstream repository and the following will be triggered:
  - creation of a Github Release
  - build of the wheels and upload them to pypi
  - once the tarball is available on pypi, the conda-forge bot will make a PR
    to the feedstock to update the conda-forge package

