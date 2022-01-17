import pathlib

import numpy as np

from pyechelle import simulator
from pyechelle.simulator import export_to_html, check_url_exists, check_for_spectrograph_model, parse_num_list, \
    available_models


def test_parse_num_list():
    assert parse_num_list("1-3") == [1, 2, 3]
    assert parse_num_list("0-7") == [0, 1, 2, 3, 4, 5, 6, 7]


def test_export_to_html():
    path_to_file = pathlib.Path(pathlib.Path.cwd().resolve()).joinpath('test.html')
    data = np.random.random((200, 200))
    export_to_html(data, path_to_file)
    print(path_to_file.is_file())
    assert path_to_file.is_file()
    path_to_file.unlink(missing_ok=True)


def test_models_exist():
    # test that URL exists for all models
    for m in available_models:
        assert check_for_spectrograph_model(m, False)

    # test download for first model
    check_for_spectrograph_model(available_models[0], True)
    assert pathlib.Path(simulator.__file__).resolve().parent.joinpath('models').joinpath(
        f"{available_models[0]}.hdf").is_file()
    pathlib.Path(simulator.__file__).resolve().parent.joinpath('models').joinpath(f"{available_models[0]}.hdf").unlink(
        missing_ok=True)


def test_check_url_exists():
    assert check_url_exists("https://pypi.org/project/pyechelle/")
    assert not check_url_exists("https://pypi.org/project/pyechelle/nonexistingurl")
