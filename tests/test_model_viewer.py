import pytest

from pyechelle import model_viewer, simulator, spectrograph


def test_model_viewer(zemax_spectrograph):
    spec = zemax_spectrograph
    model_viewer.plot_transformations(spec)
    model_viewer.plot_psfs(spec)
    model_viewer.plot_transformation_matrices(spec)


@pytest.mark.xfail(raises=NotImplementedError)
def test_model_viewer_generic_plot_transformations():
    spec = spectrograph.SimpleSpectrograph()
    model_viewer.plot_transformations(spec)
