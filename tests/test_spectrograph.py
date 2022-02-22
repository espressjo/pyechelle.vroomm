from pyechelle import simulator, spectrograph


def test_zemax_spectrograph():
    # get spectrograph if not available
    path = simulator.check_for_spectrograph_model(simulator.available_models[0])
    spec = spectrograph.ZEMAX(path)
    for o in spec.get_orders():
        min_wl, max_wl = spec.get_wavelength_range(order=o)
        assert min_wl < max_wl
