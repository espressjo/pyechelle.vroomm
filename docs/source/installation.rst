Installation
============

There are multiple ways of installing PyEchelle.
The recommended way is to install via pip.

Install via pip
---------------
The simplest way for installing pyechelle is using *pip*:

.. code-block:: bash

    pip install pyechelle



Install from source
-------------------

.. code-block:: bash

    git clone https://gitlab.com/Stuermer/pyechelle.git


After that you can either install [Poetry](https://python-poetry.org/) and use it inside the pyechelle directory to
automatically install the dependencies of PyEchelle:

.. code-block:: bash

    poetry install


or you can use pip/conda and install the required python packages that are listed in pyproject.toml
under **[tool.poetry.dependencies]** directly:
