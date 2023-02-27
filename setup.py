from setuptools import setup, find_packages
import os
from distutils.command.build_py import build_py

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists("MANIFEST"):
    os.remove("MANIFEST")

# Get __version__ from PACKAGE_NAME/__init__.py without importing the package
# __version__ has to be defined in the first line
with open("pointpats/__init__.py", "r") as f:
    exec(f.readline())

with open("README.md", "r", encoding="utf8") as file:
    long_description = file.read()


def _get_requirements_from_files(groups_files):
    groups_reqlist = {}

    for k, v in groups_files.items():
        with open(v, "r") as f:
            pkg_list = f.read().splitlines()
        groups_reqlist[k] = pkg_list

    return groups_reqlist


def setup_package():
    _groups_files = {
        "base": "requirements.txt",  # basic requirements
        "tests": "requirements_tests.txt",  # requirements for tests
        "docs": "requirements_docs.txt",  # requirements for building docs
    }
    reqs = _get_requirements_from_files(_groups_files)
    install_reqs = reqs.pop("base")
    extras_reqs = reqs

    setup(
        name="pointpats",
        version=__version__,
        description="Methods and Functions for planar point pattern analysis",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/pysal/pointpats",
        maintainer="Hu Shao",
        maintainer_email="shaohutiger@gmail.com",
        py_modules=["pointpats"],
        python_requires=">3.5",
        tests_require=["pytest"],
        keywords="spatial statistics",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: GIS",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
        ],
        license="3-Clause BSD",
        packages=find_packages(),
        install_requires=install_reqs,
        extras_require=extras_reqs,
        zip_safe=False,
        cmdclass={"build.py": build_py},
    )


if __name__ == "__main__":
    setup_package()
