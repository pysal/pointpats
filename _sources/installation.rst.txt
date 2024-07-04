.. Installation

Installation
============

From version 2.1.0, pointpats supports python `3.6`_ and `3.7`_ only.
Please make sure that you are operating in a python 3 environment.

Installing released version
---------------------------

pointpats is available on the `Python Package Index`_. Therefore, you can either
install directly with `pip` from the command line::

  pip install -U pointpats


or download the source distribution (.tar.gz) and decompress it to your selected
destination. Open a command shell and navigate to the decompressed folder.
Type::

  pip install .

You may also install the latest stable pointpats via `conda-forge`_ channel by
running::

  $ conda install --channel conda-forge pointpats

Installing development version
------------------------------

Potentially, you might want to use the newest features in the development
version of pointpats on github - `pysal/pointpats`_ while have not been incorporated
in the Pypi released version. You can achieve that by installing `pysal/pointpats`_
by running the following from a command shell::

  pip install git+https://github.com/pysal/pointpats.git

You can  also `fork`_ the `pysal/pointpats`_ repo and create a local clone of
your fork. By making changes
to your local clone and submitting a pull request to `pysal/pointpats`_, you can
contribute to the pointpats development.

.. _3.6: https://docs.python.org/3.6/
.. _3.7: https://docs.python.org/3.7/
.. _Python Package Index: https://pypi.org/project/pointpats/
.. _pysal/pointpats: https://github.com/pysal/pointpats
.. _fork: https://help.github.com/articles/fork-a-repo/
.. _conda-forge: https://github.com/conda-forge/pointpats-feedstock