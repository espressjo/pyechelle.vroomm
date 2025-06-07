import pytest
from datetime import datetime
from pyechelle import spectrograph


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    report.start = call.start
    report.stop = call.stop


@pytest.fixture(scope="package")
def MAROONX():
    """
    Creates a ZEMAX spectrograph instance that can be reused across all tests.
    """
    spec = spectrograph.ZEMAX("MaroonX")
    return spec


@pytest.fixture(scope="package")
def simple_spectrograph():
    """
    Creates a ZEMAX spectrograph instance that can be reused across all tests.
    """
    spec = spectrograph.SimpleSpectrograph()
    return spec


def pytest_terminal_summary(terminalreporter):
    terminalreporter.ensure_newline()
    terminalreporter.section("start/stop times", sep="-", bold=True)
    for stat in terminalreporter.stats.values():
        for report in stat:
            if getattr(report, "when", None) == "call":
                start = datetime.fromtimestamp(report.start)
                stop = datetime.fromtimestamp(report.stop)
                terminalreporter.write_line(
                    f"{report.nodeid:20}: {start:%Y-%m-%d %H:%M:%S} - {stop:%Y-%m-%d %H:%M:%S}"
                )
