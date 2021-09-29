import os
from io import open as io_open

from setuptools import setup, find_packages

__version__ = None
src_dir = os.path.abspath(os.path.dirname(__file__))
version_file = os.path.join(src_dir, 'abtem', '_version.py')
with io_open(version_file, mode='r') as fd:
    exec(fd.read())

fndoc = os.path.join(src_dir, 'README.md')
with io_open(fndoc, mode='r', encoding='utf-8') as fd:
    README_md = fd.read()

setup(
    name='abtem',
    version=__version__,
    description='ab initio Transmission Electron Microscopy',
    long_description=README_md,
    url='https://github.com/jacobjma/abtem',
    author='Jacob Madsen',
    author_email='jacob.madsen@univie.ac.at',
    maintainer='Jacob Madsen',
    maintainer_email='jacob.madsen@univie.ac.at',
    platforms=['any'],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.17,<=1.20',
        'scipy',
        'numba',
        'imageio',
        'pyfftw',
        'h5py',
        'matplotlib',
        'ase',
        'tqdm',
        'psutil'],
    extras_require={
        'full': ['hyperspy',
                 'pyxem']
    },
    tests_require=['pytest'],
    packages=find_packages(),
    include_package_data=True,
)
