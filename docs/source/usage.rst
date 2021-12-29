Usage
=====

Here, it is assumed that one of the provided spectrograph HDF model files is used or that the appropriate model has been
created from the ZEMAX file.

Basics
------
PyEchelle is controlled via command line arguments.

.. code-block:: bash

    pyechelle -h

lists all available arguments.

This makes it easy for PyEchelle to be scripted.

.. argparse::
   :module: pyechelle.simulator
   :func: generate_parser
   :prog: pyechelle
