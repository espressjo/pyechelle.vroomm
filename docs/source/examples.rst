Examples - command line
=======================

Example 1: Flat
^^^^^^^^^^^^^^^
As a first example, we simulate a flat field. Or more precisely, we use a source of constant spectral density.
The unit is µW/s/µm.

.. code-block:: none

    pyechelle --spectrograph MaroonX --sources Constant --constant_intensity 0.001

.. note:: Since we didn't specify the integration time manually, a default value of 1s is used.

The output will look like this:

.. raw:: html
   :file: _static/plots/example1.html

.. note:: Due to the photon-wise generation of the spectrum, all generated spectra naturally show photon noise. For a high signal to noise (S/N) spectrum, increase the integration time or the spectral density.

Example 2: multiple fields/fibers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For instruments with multiple fields or fibers, we can specify the fields with the *--fiber* keyword.
It accepts multiple arguments such as '1 3 4' for simulating fiber 1, 3 and 4.
It also accept ranges such as '2-4' and mixed range and list arguments

.. code-block:: none

    pyechelle --spectrograph MaroonX --sources Constant --constant_intensity 0.0001 --fiber 2-4

The output will look like:

.. raw:: html
   :file: _static/plots/example2.html

Example 3: A stellar source
^^^^^^^^^^^^^^^^^^^^^^^^^^^
When simulating stellar sources, a visual magnitude of the source has to be provided.
Also, a telescope size should be provided, otherwise, a default telescope of 1m diameter is used to
calculate the photon flux.
Here, we specify a telescope of 8.1m diameter of the primary mirror and 1m diameter for the secondary mirror,
a integration time of 60s.
Our source is a simulated M-dwarf spectrum using the
`PHOENIX simulations <https://www.aanda.org/articles/aa/abs/2013/05/aa19058-12/aa19058-12.html>`_ with an
effective temperature of 3500 K, Z=-1.0, alpha=0. and surface gravity of log_g=5.5. The V-band magnitude was
specified to be 14. We simulate the spectrum in the central 3 science fibers.

.. code-block:: none

    pyechelle --spectrograph MaroonX --sources Phoenix --phoenix_t_eff 3500 --phoenix_z -1.0 --phoenix_alpha 0. --phoenix_log_g 5.5 --phoenix_magnitude 14 --fiber 2-4 -t 60

.. raw:: html
   :file: _static/plots/example3.html


Example 4: Multiple sources and readnoise
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We can simulate different sources for the different fields to realize e.g. a simultaneous calibration spectrum.
In order to pass multiple sources, we have to pass multiple arguments to *--sources* and
the number of the arguments has to match the number of fields.


.. code-block:: none

    pyechelle --spectrograph MaroonX --sources Etalon Phoenix Phoenix Phoenix --phoenix_t_eff 3500 --phoenix_z -1.0 --phoenix_alpha 0. --phoenix_log_g 5.5 --phoenix_magnitude 14 --fiber 1-4 --d_primary 8.1 --d_secondary 1 -t 30 --etalon_n_photons 1000 --etalon_d 10 --bias 1000 --read_noise 3


.. raw:: html
   :file: _static/plots/example4.html


Have fun !