from setuptools import setup, find_packages

setup(
    name='abtem',
    version='0.1',
    description='abtem',
    author='Jacob Madsen',
    author_email='jacob.madsen@univie.ac.at',
    packages=find_packages(),  # same as name
    include_package_data=True,
    install_requires=[],
)
