"""
PyBEC
Python package for extracting and manipulating Born Effective Charges from QuantumEspresso Output
"""
import sys
from setuptools import setup, find_packages
import versioneer

short_description = __doc__.split("\n")

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except FileNotFoundError:
    long_description = "\n".join(short_description[2:])


setup(
    # Self-descriptive entries which should always be present
    name='pybec',
    author='Brett Henderson',
    author_email='brettrhenderson25@gmail.com',
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='MIT',

    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),

    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,

    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,

    # Additional entries you may want simply uncomment the lines you want and fill in the data
    url='https://pybec.readthedocs.io/en/latest/',  # Website
    install_requires=[    # Required packages, pulls from pip if needed; do not use for Conda deployment
        'toolz>=0.8.2',
        'dask>=1.2.2',
        'matplotlib>=3.0.3',
        'numpy>=1.16.2',
        'pandas>=0.24.2',
        'scipy>=1.2.1',
        'sklearn',
        'pykrige'
    ],
    platforms=['Linux',
               'Mac OS-X',
               'Unix',
               'Windows'],            # Valid platforms your code works on, adjust to your flavor
    python_requires=">=3.5",          # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,

)
