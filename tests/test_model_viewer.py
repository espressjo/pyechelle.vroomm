import pytest

from pyechelle import model_viewer


def test_model_viewer(MAROONX):
    model_viewer.plot_transformations(MAROONX)
    model_viewer.plot_psfs(MAROONX)
    model_viewer.plot_transformation_matrices(MAROONX)


@pytest.mark.xfail(raises=NotImplementedError)
def test_model_viewer_generic_plot_transformations(simple_spectrograph):
    model_viewer.plot_transformations(simple_spectrograph)
