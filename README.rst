SPMLIB: Sparse Modeling Library
===============================
SPMLIB is a Python package for sparse modeling and optimization.


Installation
------------

To install the latest version from `GitHub <https://github.com/tsakailab/spmlib>`_ do

::

    git clone https://github.com/tsakailab/spmlib.git
    cd spmlib
    python setup.py install

The install commands will have to be performed with sudo.

Another choice for a developer is to install as follows:

::

    python setup.py develop --user

Then you can edit the source codes in this package and see directly the changes without reinstallation.
In this case, you can easily uninstall by

::

    python setup.py develop -u

This developer installation is recommended before we register SPMLIB at PyPI to enable pip installation.
