How to create a new spectrograph model
======================================

Here, we describe how to create a new model file from an existing ZEMAX design.

Make a copy
-----------
Since we are modifying the ZEMAX file, the first step always is to save the ZEMAX file under a new name.

Check apertures & PSF settings
------------------------------
When tracing, it is important that the rays end up on the last surface which is assumed to be the detector, rather
than being vignetted somewhere else. In almost all cases, vignetting on surfaces prior to the detector is an indication
that the ZEMAX model should be revised. To avoid vignetting, it's typically a good idea to remove all apertures except
the one that defines the detector. The aperture on the last surface is used to determine the wavelength bounds for each
diffraction order.

The code will warn you in case that during tracing, the rays are vignetted on a
surface other than the detector.

Within ZEMAX, you should also adjust the PSF settings to something reasonable. Set the image delta to something like 1/3
of the CCD pixel size and make the image sampling big enough to fully capture the PSF.
Again, the code will warn you if the PSF spills over the sampled area, but it's better to avoid this in the first
place and check manually at a few wavelengths.

Use InteractiveZEMAX spectrograph + HDFBuilder
----------------------------------------------
Once the ZEMAX file is prepared, you can use the InteractiveZEMAX spectrograph
in conjunction with the HDFBuilder.

The following example shows how to then generate .HDF spectrograph models that
can later be used independently of ZEMAX for simulations.

.. code-block:: python

    from pyechelle.CCD import CCD
    from pyechelle.hdfbuilder import HDFBuilder
    from pyechelle.spectrograph import InteractiveZEMAX

    # Open Link to a standalone OpticStudio instance
    zmx = InteractiveZEMAX(name='AWESOME_SPECTROGRAPH', zemax_filepath="PATH_TO_ZMX/ZOS_FILE")

    # set basic grating specifications
    zmx.set_grating(surface='echelle', blaze=-75.964, theta=0., gamma=-0.75)

    # add CCD information (only one CCD supported so far. So for instruments with multiple CCDs, you have to generate
    # separate models for now.
    zmx.add_ccd(1, CCD(10560, 10560, pixelsize=9))

    # Add here as many fiber/fields as you wish. You don't have to fiddle with the fields in OpticStudio. The
    # existing fields will be ignored/deleted.
    zmx.add_field(0., 0., 75 * 4, 75, shape='rectangular', name='Science fiber')

    # Add here a list with the diffraction orders you want to include
    zmx.set_orders(1, 1, list(range(65, 100)))

    # Adjust settings for the Huygens PSF. Best to check out 'reasonable' parameters manually in ZEMAX first.
    zmx.psf_settings(image_delta=0.4, image_sampling="64x64", pupil_sampling="128x128")

    # at this point, you can interact with the spectrograph interactively, e.g. by doing something like:
    # zmx.get_psf(wavelength=0.72, order=85, fiber=1, ccd_index=1)
    # zmx.get_wavelength_range(order=85)
    # zmx.get_transformation(wavelength=0.72, order=85, fiber=1, ccd_index=1)
    # and it will/should pull the appropriate values from ZEMAX. This might be helpful for debugging.

    # To generate an .HDF model file, you do:
    hdf = HDFBuilder(zmx, '../pyechelle/models/AWESOME_SPECTROGRAPH.hdf')
    # this will take a long time...
    hdf.save_to_hdf(n_transformation_per_order=50, n_psfs_per_order=5)