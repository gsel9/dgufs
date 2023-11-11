# -*- coding: utf-8 -*-
#
# setup.py
#
# This module is part of dgufs
#

"""
Setup script for dgufs package.
"""

__author__ = "Severin Elvatun"
__email__ = "langberg91@gmail.com"


from setuptools import find_packages, setup

PACKAGE_NAME = "dgufs"
VERSION = "1.0.0"
KEYWORDS = "unsupervised feature selection, machine learning, data analysis"
TESTS_REQUIRE = ["pytest", "mock", "pytest_mock"]


def readme():
    """Return the contents of the README.md file."""

    with open("README.md") as freadme:
        return freadme.read()


def requirements():
    """Return the contents of the REQUIREMENTS.txt file."""

    with open("REQUIREMENTS.txt", "r") as freq:
        return freq.read().splitlines()


def proj_license():
    """Return the contents of the LICENSE.txt file."""

    with open("LICENSE.txt") as flicense:
        return flicense.read()


setup(
    author="Severin Elvatun",
    author_email="langberg91@gmail.com",
    description="A Python implementation of the The Dependence Guided "
    "Unsupervised Feature Selection algorithm by Jun Guo and "
    "Wenwu Zhu (2018).",
    url="https://github.com/gsel9/dgufs",
    install_requires=requirements(),
    long_description=readme(),
    license=proj_license(),
    name=PACKAGE_NAME,
    version=VERSION,
    packages=find_packages(exclude=["test", "tests.*"]),
    setup_requires=["pytest-runner"],
    tests_require=TESTS_REQUIRE,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
)
