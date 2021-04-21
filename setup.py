#!/usr/bin/env python3

import setuptools
import os

package_name = "mtftorch"
packages = setuptools.find_packages(
    include=[package_name, "{}.*".format(package_name)]
)

# Version info -- read without importing
_locals = {}
with open(os.path.join(package_name, "version.py")) as fp:
    exec(fp.read(), None, _locals)
version = _locals["__version__"]
binary_names = _locals["binary_names"]

# Constant known variables used throughout this file
cwd = os.path.dirname(os.path.abspath(__file__))

# Read in README.md for our long_description
with open(os.path.join(cwd, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name=package_name,
    version=version,
    description="PyTorch-style interface to Mesh Tensorflow",
    license="BSD",
    long_description=long_description,
    author="Shawn Presser",
    author_email="shawnpresser@gmail.com",
    url="https://github.com/shawwn/mtftorch",
    install_requires=[
        # 'Click>=7.1.2',
        # 'six>=1.11.0',
        # 'ring>=0.7.3',
        # 'moment>=0.0.10',
        # 'google-auth>=0.11.0',
        # 'google-api-python-client>=1.7.11',
        'mesh_tensorflow>=0.1.19',
    ],
    packages=packages,
    entry_points={
        "console_scripts": [
            "{} = {}.program:cli".format(binary_name, package_name)
            for binary_name in binary_names
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
    ],
)

