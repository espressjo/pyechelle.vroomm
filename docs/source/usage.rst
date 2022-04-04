Usage
=====

There are two ways of using PyEchelle: either via the command line interface or via python.
The command line interface only works for spectrograph models that are available as HDF models (see :ref:`Models and their perturbation` for more
details).

Python interface
----------------
The python interface allows to write simulations scripts in python.
A minimal example looks like this:

.. literalinclude:: ../../examples/00_minimal_example.py
  :language: python
  :linenos:
  :start-after: "__main__":
  :end-at: sim.run()
  :tab-width: 0
  :dedent: 4

For further examples see :ref:`Examples - python scripts`

Command line interface
----------------------
When installing PyEchelle via pip, the command 'pyechelle' becomes available:

.. code-block:: bash

    pyechelle -h

lists all available arguments.

This makes it easy for PyEchelle to be scripted.

.. argparse::
   :module: pyechelle.simulator
   :func: generate_parser
   :prog: pyechelle
