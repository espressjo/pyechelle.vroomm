from pyechelle import model_viewer, simulator, spectrograph


def test_model_viewer():
    path = simulator.check_for_spectrogrpah_model(simulator.available_models[0])
    spec = spectrograph.ZEMAX(path, 1)
    model_viewer.plot_transformations(spec)
    model_viewer.plot_psfs(spec)
    model_viewer.plot_transformation_matrices(spec)
