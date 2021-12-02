from pyechelle.simulator import parse_num_list


def test_parse_num_list():
    assert parse_num_list("1-3") == [1, 2, 3]
    assert parse_num_list("0-7") == [0, 1, 2, 3, 4, 5, 6, 7]