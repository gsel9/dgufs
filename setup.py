# stdlib
import os

# third party
from setuptools import setup

PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def read(fname: str) -> str:
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_version() -> str:
    return "0.0.1"


if __name__ == "__main__":
    try:
        setup(
            version=get_version(),
            author="Severin Elvatun",
            author_email="langberg91@gmail.com",
            description="""A Python implementation of the The Dependence Guided \
                           Unsupervised Feature Selection algorithm \
                           by Jun Guo and Wenwu Zhu (2018).""",
            long_description=read("README.md"),
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise


# from setuptools import find_packages, setup

# PACKAGE_NAME = "dgufs"
# VERSION = "1.0.1"
# KEYWORDS = "unsupervised feature selection, machine learning, data analysis"
# TESTS_REQUIRE = ["pytest", "pytest-cov", "mock", "pytest_mock"]


# setup(
#    author="Severin Elvatun",
#    author_email="langberg91@gmail.com",
#    url="https://github.com/gsel9/dgufs",
#    install_requires=requirements(),
#    long_description=readme(),
#    license=proj_license(),
#    name=PACKAGE_NAME,
#    version=VERSION,
#    packages=find_packages(exclude=["test", "tests.*"]),
# py_modules=["dgufs"],
#    setup_requires=["pytest-runner"],
#    tests_require=TESTS_REQUIRE,
#    classifiers=[
#        "Development Status :: 3 - Alpha",
#        "Intended Audience :: Science/Research",
#        "Environment :: Console",
#        "Programming Language :: Python :: 3",
#        "Programming Language :: Python :: 3.5",
#        "Programming Language :: Python :: 3.6",
#    ],
# )
