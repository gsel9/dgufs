# -*- coding: utf-8 -*-
#
# setup.py
#
# This module is part of dgufs
#

"""
Setup script for dgufs package.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'langberg91@gmail.no'


from setuptools import setup, find_packages


PACKAGE_NAME = 'dgufs'
VERSION = '0.1.0'
KEYWORDS = 'unsupervised feature selection, machine learning, data analysis'
TESTS_REQUIRE = ['pytest', 'mock', 'pytest_mock']


def readme(_):
    """Return the contents of the README.md file."""

    with open('README.md') as freadme:
        return freadme.read()


def requirements(_):
    """Return the contents of the REQUIREMENTS.txt file."""

    with open('REQUIREMENTS.txt', 'r') as freq:
        return freq.read().splitlines()


def license(_):
    """Return the contents of the LICENSE.txt file."""

    with open('LICENSE.txt') as flicense:
        return flicense.read()


setup(
    author="Severin Langberg",
    author_email="langberg91@gmail.com",
    description="A Python implementation of the The Dependence Guided "
                "Unsupervised Feature Selection algorithm by Jun Guo and "
                "Wenwu Zhu (2018).",
    url='https://github.com/GSEL9/dgufs',
    install_requires=requirements(None),
    long_description=readme(None),
    license=license(None),
    name=PACKAGE_NAME,
    version=VERSION,
    packages=find_packages(exclude=['test', 'tests.*']),
    setup_requires=['pytest-runner'],
    tests_require=TESTS_REQUIRE,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Environment :: Console',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
)
