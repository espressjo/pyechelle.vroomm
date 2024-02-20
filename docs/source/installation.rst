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


After that, the easiest way to install the package is to first install `Poetry <https://python-poetry.org/>`_ and use it inside the pyechelle directory to
automatically install the dependencies of PyEchelle:

.. code-block:: bash

    poetry install


Alternatively, you have to install the dependencies that are listed in pyproject.toml
under **[tool.poetry.dependencies]** manually.

