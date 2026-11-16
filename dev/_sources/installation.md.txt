# Installation

Currently, pointpats supports Python >= [3.12](https://docs.python.org/3.12/). Please
make sure that you are operating in a Python 3 environment.

## Installing released version

pointpats is available on the
[Python Package Index](https://pypi.org/project/pointpats/). Therefore, you can either
install directly with `pip` from the command line:

```sh
pip install -U pointpats
```

or download the source distribution (.tar.gz) and decompress it to your selected
destination. Open a command shell and navigate to the decompressed folder. Type:

```sh
pip install .
```

You may also install the latest stable pointpats via
[conda-forge](https://github.com/conda-forge/pointpats-feedstock) channel by running:

```sh
conda install -c conda-forge pointpats
```

## Installing development version

Potentially, you might want to use the newest features in the development version of
pointpats on GitHub - [pysal/pointpats](https://github.com/pysal/pointpats) while have
not been incorporated in the Pypi released version. You can achieve that by installing
[pysal/pointpats](https://github.com/pysal/pointpats) by running the following from a
command shell:

```sh
pip install git+https://github.com/pysal/pointpats.git
```

You can also [fork](https://help.github.com/articles/fork-a-repo/) the
[pysal/pointpats](https://github.com/pysal/pointpats) repo and create a local clone of
your fork. By making changes to your local clone and submitting a pull request to
[pysal/pointpats](https://github.com/pysal/pointpats), you can contribute to the
pointpats development.