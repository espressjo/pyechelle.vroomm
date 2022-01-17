from pyechelle import simulator, spectrograph


def test_zemax_spectrograph():
    # get spectrograph if not available
    path = simulator.check_for_spectrograph_model(simulator.available_models[0])
    spec = spectrograph.ZEMAX(path, 1)
    for o in spec.orders:
        min_wl, max_wl = spec.get_wavelength_range(fiber=o)
        assert min_wl < max_wl
